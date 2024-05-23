from dataclasses import dataclass
from os.path import basename, dirname, exists, splitext
from typing import Collection, Literal

import faiss
import numpy as np
import pandas as pd
import pyfastx
import torch
import torch.distributed as dist
from lightning.pytorch.callbacks import BasePredictionWriter
from torch.utils.data import Dataset, get_worker_info
from transformers import AutoModel, AutoTokenizer


## ------------------------------------------------------------------------------
## Utility Classes and factories
## ------------------------------------------------------------------------------
@dataclass
class BedRegion:
    # genomic interval
    chrom: str
    start: int
    end: int

    # Used to associate faiss index vectors with the region
    id: int

    def __repr__(self):
        return f"{self.chrom}\t{self.start}\t{self.end}\t{self.id}"


def parse_bed(bedfile: str) -> list[BedRegion]:
    regions = []
    with open(bedfile, "r") as f:
        for line in f:
            chrom, start, end, id = line.rstrip().split()
            regions.append(BedRegion(chrom, int(start), int(end), int(id)))
    return regions


## ------------------------------------------------------------------------------
## Utility functions
## ------------------------------------------------------------------------------
def load_model(checkpoint: str) -> tuple[AutoModel, AutoTokenizer]:
    model = AutoModel.from_pretrained(
        checkpoint, device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    return model, tokenizer


def tokenize_batch(batch: list[str], tokenizer) -> torch.Tensor:
    return torch.LongTensor(
        # TODO give option for upper or lower case
        tokenizer([s.upper() for s in batch], padding="longest")["input_ids"]
    )


## ------------------------------------------------------------------------------
## Inference writers
## ------------------------------------------------------------------------------
class FaissIndexWriter(BasePredictionWriter):
    """
    Given a output path, write the predictions to a faiss index.
    """

    def __init__(
        self,
        index_path: str,
        dim: int,
        with_ids: bool = False,
    ):
        super().__init__(write_interval="batch")
        self.index_path = index_path
        self.index = faiss.IndexFlatL2(dim)
        self.with_ids = with_ids

        if with_ids:
            self.index = faiss.IndexIDMap(self.index)

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        # put all predictions across all devices into one tensor
        gathered_embeddings = torch.zeros(
            (dist.get_world_size() * prediction.shape[0], *prediction.shape[1:]),
            device=prediction.device,
        )
        dist.all_gather_into_tensor(gathered_embeddings, prediction)

        if self.with_ids:
            gatherered_ids = [None] * dist.get_world_size()
            dist.all_gather_object(gatherered_ids, batch["id"])
            gathered_ids = np.concatenate(gatherered_ids)

        dist.barrier()  # synchronize

        # only the global zero process writes to the index
        if not trainer.is_global_zero:
            return
        if self.with_ids:
            # last batch padding issue with DDP
            if len(gathered_ids) != gathered_embeddings.shape[0]:
                self.index.add_with_ids(
                    gathered_embeddings.cpu().numpy()[: len(gathered_ids)], gathered_ids
                )
            else:
                self.index.add_with_ids(gathered_embeddings.cpu().numpy(), gathered_ids)
        else:
            # TODO its unclear wether DDP padding is an issue here
            # It likely has a miniscule effect on the index.
            # perhaps in the future I will only make indices `with_ids`
            self.index.add(gathered_embeddings.cpu().numpy())

    def on_predict_epoch_end(self, trainer, pl_module):
        dist.barrier()
        if not trainer.is_global_zero:
            return
        faiss.write_index(self.index, self.index_path)


class KNNReferenceQueryWriter(BasePredictionWriter):
    def __init__(self, index, id_bed, topk, output):
        super().__init__(write_interval="batch")
        # self.index = faiss.read_index(index)
        self.topk = topk

        # we'll be writing in append mode so create/clear the file first
        self.output = output
        with open(self.output, "w") as f:
            pass

        # TODO check to make sure that the ids are in order
        self.id_bed = id_bed
        self.bed_regions = np.array(parse_bed(id_bed), dtype=object)

    def setup(self, trainer, pl_module, stage):
        if not trainer.is_global_zero:
            return
        # don't load the index needlessly
        self.index = faiss.read_index(self.index)

    def write_results(self, out, fnames, names, regions, D):
        """
        Write a batch of results to the output file.
        """
        # NOTE: we side step the ddp batch padding issue because of the zip
        for f, n, r, d in zip(fnames, names, regions, D):
            out.write("\t".join([f, n, f"{r.chrom}:{r.start}-{r.end},{d}"]))

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        ## combine results/batches ---------------------------------------------
        gathered_embeddings = torch.zeros(
            (dist.get_world_size() * prediction.shape[0], *prediction.shape[1:]),
            device=prediction.device,
        )
        dist.all_gather_into_tensor(gathered_embeddings, prediction)

        gathered_batches = [None] * dist.get_world_size()
        dist.all_gather_object(gathered_batches, batch)
        filenames = [f for fnames in gathered_batches["filename"] for f in fnames]
        names = [n for names in gathered_batches["name"] for n in names]

        dist.barrier()
        if not trainer.is_global_zero:
            return

        ## search and write results --------------------------------------------
        D, I = self.index.search(gathered_embeddings.cpu().numpy(), self.topk)
        regions = self.bed_regions[I]
        with open(self.output, "a") as out:
            self.write_results(out, filenames, names, regions, D)


class FaissQueryWriter(BasePredictionWriter):
    """
    Use predictions as queries to search the faiss index.
    """

    # TODO rename this to something more specific to the task
    # or make it more general to handle any kind of query and output schema

    def __init__(
        self,
        index_path: str,
        topk: int,
        output: str,
        clear_existing: bool = False,
    ):
        super().__init__(write_interval="batch")
        self.index = faiss.read_index(index_path)
        self.topk = topk
        self.output = output

        if clear_existing:
            with open(self.output, "w") as f:
                pass
        else:
            raise ValueError(
                "Output file already exists.  Set clear_existing=True to overwrite."
            )

    def write_scores(self, samples, positions, D, f):
        for s, p, d in zip(samples, positions, D):
            score = np.mean(d)
            f.write(f"{s}\t{p[0]}\t{p[1]}\t{score}\n")

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        """
        - prediction: tensor containing batch of embedding vectors
        - batch: dict of batch data with keys
            - "sample": list of sample names
            - "pos": list of start/end tuples
            - "seq": tokenized sequences
        """
        gathered_preds = [None] * dist.get_world_size()
        dist.all_gather_object(gathered_preds, prediction)

        gathered_batches = [None] * dist.get_world_size()
        dist.all_gather_object(gathered_batches, batch)

        dist.barrier()
        if not trainer.is_global_zero:
            return

        with open(self.output, "a") as f:
            for gp, gb in zip(gathered_preds, gathered_batches):
                samples = [sample for sample in gb["sample"]]
                positions = [pos for pos in gb["pos"]]
                D, _ = self.index.search(gp.cpu().numpy(), self.topk)
                self.write_scores(samples, positions, D, f)


## ------------------------------------------------------------------------------
## Datasets
## ------------------------------------------------------------------------------
class FastqDataset(Dataset):
    """
    Dataset for loading a fastq file using pyfastx and num_workers > 0.
    You have to use the provided worker_init_fn to load the pyfastx object in each
    worker during dataloader initialization.

    example:
    dataset = FastqDataset(filename, remake_index=True)
    dataloader = DataLoader(
        dataset,
        num_workers=8,
        batch_size=64,
        worker_init_fn=FastqDataset.worker_init_fn,
        collate_fn=functools.partial(dataset.collate_fn, tokenizer=tokenizer),
    )
    """

    def __init__(self, filename, remake_index=False):
        self.filename = filename
        if not exists(filename + ".fxi") or remake_index:
            self.make_index()

    def make_index(self):
        """build the index before initializing workers."""
        pyfastx.Fastq(self.filename, build_index=True)

    def __len__(self):
        fastq = pyfastx.Fastq(self.filename)
        return len(fastq)

    def __getitem__(self, idx):
        return self.fastq[idx]

    def collate_fn(self, batch, tokenizer):
        return {
            "filename": [self.filename for _ in batch],
            "seq": tokenize_batch([x.seq for x in batch], tokenizer),
            "name": [x.name for x in batch],
            "qual": [x.qual for x in batch],
        }

    @staticmethod
    def worker_init_fn(worker_id):
        worker_info = get_worker_info()
        dataset = worker_info.dataset
        fastq = pyfastx.Fastq(dataset.filename)
        dataset.fastq = fastq


class SiameseDataset(Dataset):
    """
    Load tab separated file with columns A, B, sim
    where A, B are sequences and sim is a float similarity score.
    """

    def __init__(self, data: str, tokenizer: AutoTokenizer):
        # We're assuming the data fits in memory.
        # Otherwise we'd need to use some kind of
        # lazy loading alternative to the dataframe.
        self.data = pd.read_csv(data, sep="\t", names=["A", "B", "sim"])
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx]

    def collate_fn(self, batch: list[dict]):
        A, B, sim = zip(*[(x["A"], x["B"], x["sim"]) for x in batch])

        return {
            "A": tokenize_batch(A, self.tokenizer),
            "B": tokenize_batch(B, self.tokenizer),
            "sim": torch.FloatTensor(sim),
        }


class SlidingWindowReferenceFasta(Dataset):
    """
    Take a reference genome (fasta) and generate sliding windows of sequences.
    """

    def __init__(
        self,
        fasta_path: str,
        window_size: int,
        stride: int,
        chromosome_names: Collection[str],
        letter_case: Literal["upper", "lower"] = "upper",
    ):
        self.working_dir = dirname(fasta_path)
        self.fasta_path = fasta_path
        self.fasta_index = f"{fasta_path}.fai"
        if not exists(fasta_index := f"{fasta_path}.fai"):
            raise FileNotFoundError(f"Index file not found: {fasta_index}")

        self.window_size = window_size
        self.stride = stride
        self.chromosome_names = set(chromosome_names)
        self.bed_file = fasta_path + ".bed"

        ## Make sliding window bed file to aid in retrieving windows of fasta sequence
        # we will write this file to disk for later use,
        # and load it into memory now for use in the dataset.
        if not exists(self.bed_file):
            self.chromosome_lengths = self._get_chromosome_lengths()
            print(f"Making strided bed file for {fasta_path}.")
            self._make_strided_bed(f"{self.bed_file}")
        self.bed_regions = parse_bed(self.bed_file)

        ## Load the reference sequences into memory stored as a dict keyed by chromosome name
        # additionally we map the sequences to the desired case.  This is important for
        # different tokenizers. eg heynadna uses uppercase letters only
        print(f"Loading reference sequences from {fasta_path}.")
        if letter_case == "lower":
            self.case_mapping = str.lower
        elif letter_case == "upper":
            case_mapping = str.upper
        else:
            raise ValueError(f"Invalid letter case: {letter_case}")

        fasta = pyfastx.Fasta(fasta_path)
        self.sequences = {
            chrom: case_mapping(fasta[chrom].seq) for chrom in self.chromosome_names
        }

    def _get_chromosome_lengths(self) -> dict[str, int]:
        """
        Read fai to get the chromosome lengths for the provided chromosomes names.
        """
        with open(self.fasta_index, "r") as f:
            return {
                x[0]: int(line.split("\t")[1])
                for line in f
                if (x := line.split("\t"))[0] in self.chromosome_names
            }

    def _make_strided_bed(self, outfile: str):
        """
        Generate a bed file for a set of chromosomes with a sliding window.
        Then Load the bed file into memory.
        - outfile: path to write the bed intervals
        - chroms: dict of chromosome lengths
        - window_size: size of the window
        - stride: stride of the sliding window
        """
        with open(outfile, "w") as f:
            for chrom, length in self.chromosome_lengths.items():
                # write bed intervals along with a unique int id
                for i, start in enumerate(
                    range(0, length - self.window_size + 1, self.stride)
                ):
                    end = start + self.window_size
                    f.write(f"{chrom}\t{start}\t{end}\t{i}\n")

    def __getitem__(self, idx: int):
        r = self.bed_regions[idx]
        return {
            "seq": self.sequences[r.chrom][r.start : r.end + 1],
            "chrom": r.chrom,
            "start": r.start,
            "end": r.end,
            "id": r.id,
        }

    def __len__(self):
        return len(self.bed_regions)

    @staticmethod
    def collate_fn(batch: list[dict], tokenizer):
        return {
            "seq": tokenize_batch([x["seq"] for x in batch], tokenizer),
            "chrom": [x["chrom"] for x in batch],
            "start": [x["start"] for x in batch],
            "end": [x["end"] for x in batch],
            "id": [x["id"] for x in batch],
        }


class SlidingWindowFasta(Dataset):
    """
    Given a fasta with a single sequence, generate sliding windows
    of a fixed size and stride.  Additionally provide the start/end
    positions of the windows.
    """

    def __init__(self, fasta_path, window_size, stride):
        fasta = pyfastx.Fasta(fasta_path)
        self.sample_name = splitext(basename(fasta_path))[0]
        self.name = fasta[0].name.split()[0]
        self.sequence = fasta[0].seq.upper()

        self.windowed_sequences, self.positions = self.sliding_window(
            self.sequence, window_size, stride
        )

    def sliding_window(self, sequence: str, window_size: int, stride: int):
        """
        Return a list of windowed sequences and their start and end indices
        """
        return (
            [
                sequence[i : i + window_size]
                for i in range(0, len(sequence) - window_size + 1, stride)
            ],
            [
                (i, i + window_size - 1)
                for i in range(0, len(sequence) - window_size + 1, stride)
            ],
        )

    def __len__(self):
        return len(self.windowed_sequences)

    def __getitem__(self, idx):
        return {
            "seq": self.windowed_sequences[idx],
            "pos": self.positions[idx],
            "sample": self.sample_name,
        }

    @staticmethod
    def collate_fn(batch: list[dict], tokenizer):
        return {
            "seq": tokenize_batch([x["seq"] for x in batch], tokenizer),
            "pos": [x["pos"] for x in batch],
            "sample": [x["sample"] for x in batch],
        }


class SequenceDataset(Dataset):
    """
    Take a list of sequences and just serve them up as a dataset.
    """

    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]
