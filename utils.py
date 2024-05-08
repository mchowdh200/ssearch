from os.path import basename, splitext

import faiss
import numpy as np
import pandas as pd
import pyfastx
import torch
import torch.distributed as dist
from lightning.pytorch.callbacks import BasePredictionWriter
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer


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
    return torch.LongTensor(tokenizer(batch, padding="longest")["input_ids"])


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
    ):
        super().__init__(write_interval="batch")
        self.index_path = index_path
        self.index = faiss.IndexFlatL2(dim)

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
        # put all predictions across all devices into one process

        # gathered = [None] * dist.get_world_size()
        gathered = torch.zeros(
            (dist.get_world_size() * prediction.shape[0], *prediction.shape[1:]),
            device=prediction.device,
        )
        dist.all_gather_into_tensor(gathered, prediction)
        dist.barrier()

        # only the global zero process writes to the index
        if not trainer.is_global_zero:
            return
        self.index.add(gathered.cpu().numpy())

    def on_predict_epoch_end(self, trainer, pl_module):
        dist.barrier()
        if not trainer.is_global_zero:
            return
        faiss.write_index(self.index, self.index_path)


class FaissQueryWriter(BasePredictionWriter):
    """
    Use predictions as queries to search the faiss index.
    """

    # TODO make this class more general
    # for example: what if we only provide the batch of sequences?
    # Could possibly add some sort of batch schema to the writer

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
## Training datasets
## ------------------------------------------------------------------------------
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


## ------------------------------------------------------------------------------
## Inference datasets
## ------------------------------------------------------------------------------
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
        self.sequence = fasta[0].seq

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
