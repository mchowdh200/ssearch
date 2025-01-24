import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from os.path import exists
from pathlib import Path
from typing import Literal

import pandas as pd
import pyfastx
import torch
from torch.utils.data import Dataset, get_worker_info
from transformers import AutoTokenizer


class LenDataset(ABC, Dataset):
    """
    Abstract class for datasets that have a __len__ to satisfy the type checker...
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def __len__(self) -> int: ...


def get_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


class SiameseDataset(LenDataset):
    """
    Load tab separated file with columns A, B, sim
    where A, B are sequences and sim is a float similarity score.
    """

    def __init__(self, data: str, base_model: str, upper_case: bool = True):
        # We're assuming the data fits in memory.
        # Otherwise we'd need to use some kind of
        # lazy loading alternative to the dataframe.
        self.data = pd.read_csv(data, sep="\t", names=["A", "B", "sim"])
        self.tokenizer = get_tokenizer(base_model)
        self.pad_token_id = self.tokenizer.pad_token_id
        self.base_model = base_model
        self.upper_case = upper_case

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx]

    def tokenize_batch(self, batch: list[str]):
        return self.tokenizer.batch_encode_plus(
            [b.upper() for b in batch] if self.upper_case else batch,
            return_tensors="pt",
            padding="longest",
        )["input_ids"]

    def collate_fn(self, batch: list[dict]):
        A, B, sim = zip(*[(x["A"], x["B"], x["sim"]) for x in batch])
        A_ids = self.tokenize_batch(A)
        B_ids = self.tokenize_batch(B)
        A_mask = torch.where(A_ids != self.pad_token_id, True, False)
        B_mask = torch.where(B_ids != self.pad_token_id, True, False)
        sim = torch.FloatTensor(sim)

        return {
            "A": A_ids,
            "B": B_ids,
            "A_mask": A_mask,
            "B_mask": B_mask,
            "sim": sim,
        }


class FastxDataset(LenDataset):
    """
    Dataset for loading a fastq/fasta file using pyfastx and num_workers > 0.
    You have to use the provided worker_init_fn to load the pyfastx object in each
    worker during dataloader initialization.

    example:
    dataset = FastxDataset(filename, remake_index=True)
    dataloader = DataLoader(
        dataset,
        num_workers=8,
        batch_size=64,
        worker_init_fn=FastxDataset.worker_init_fn,
        collate_fn=dataset.collate_fn,
    )
    """

    def __init__(
        self,
        filename: str,
        file_type: Literal["fasta", "fastq"],
        remake_index=False,
    ):
        self.filename = filename
        self.file_type = file_type

        # we need this because Fastx does not support random access or len
        self.fastx_fun = pyfastx.Fasta if file_type == "fasta" else pyfastx.Fastq

        if not exists(filename + ".fxi") or remake_index:
            self.make_index()

    def make_index(self):
        """build the index before initializing workers."""
        self.fastx_fun(self.filename, build_index=True)

    @staticmethod
    def tokenize_batch(batch, tokenizer, upper_case):
        return tokenizer.batch_encode_plus(
            [b.upper() for b in batch] if upper_case else batch,
            return_tensors="pt",
            padding="longest",
        )["input_ids"]

    def __len__(self):
        fastx = self.fastx_fun(self.filename)
        return len(fastx)

    def __getitem__(self, idx):
        return self.fastx[idx]

    @staticmethod
    def basic_collate_fn(batch, tokenizer, upper_case):
        """
        Collate function that returns tokenized sequences and attention masks
        """
        input_ids = FastxDataset.tokenize_batch(
            [x.seq for x in batch], tokenizer, upper_case
        )
        attention_mask = torch.where(input_ids != tokenizer.pad_token_id, True, False)
        return {
            "input_ids": input_ids,
            "name": [x.name for x in batch],
            "attention_mask": attention_mask,
        }

    @staticmethod
    def worker_init_fn(worker_id):
        worker_info = get_worker_info()
        dataset = worker_info.dataset
        fastx = dataset.fastx_fun(dataset.filename)
        dataset.fastx = fastx


class SlidingWindowContig(LenDataset):
    """
    I was too lazy to generalize the sliding window fasta dataset
    so here we are...
    """

    def __init__(
        self,
        fasta: str,
        contig: str,
        window_size: int,
        stride: int,
    ):
        self.filename = fasta
        self.contig = contig
        # self.fasta = pyfastx.Fasta(fasta, build_index=True)
        # self.sequence = self.fasta[contig]
        self.window_size = window_size
        self.stride = stride
        # self.seq, self.antisense, self.positions = self.sliding_windows()
        # self.name = self.sequence.name

    @staticmethod
    def worker_init_fn(worker_id):
        # I hate doing this...
        worker_info = get_worker_info()
        dataset = worker_info.dataset
        dataset.fasta = pyfastx.Fasta(dataset.filename)
        dataset.sequence = dataset.fasta[dataset.contig]
        dataset.seq, dataset.antisense, dataset.positions = dataset.sliding_windows()
        dataset.name = dataset.sequence.name

    def sliding_windows(self):
        windows = [
            self.sequence[i : i + self.window_size]
            for i in range(0, len(self.sequence) - self.window_size + 1, self.stride)
        ]
        positions = [
            (i, i + self.window_size - 1)
            for i in range(0, len(self.sequence) - self.window_size + 1, self.stride)
        ]

        seq = [w.seq.upper() for w in windows]
        antisense = [w.antisense.upper() for w in windows]

        return seq, antisense, positions

    @staticmethod
    def tokenize_batch(batch: list[str], tokenizer):
        return tokenizer.batch_encode_plus(
            batch,
            return_tensors="pt",
            padding="longest",
        )["input_ids"]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return {
            "seq": self.seq[idx],
            "antisense": self.antisense[idx],
            "pos": self.positions[idx],  # start and end positions
            "name": self.name,  # it'll be the chrom + other stuff
        }

    @staticmethod
    def collate_fn(batch, tokenizer):
        # NOTE: this essentially doubles the batch size
        seq = [x["seq"] for x in batch]
        antisense = [x["antisense"] for x in batch]

        seq_ids = SlidingWindowContig.tokenize_batch(seq, tokenizer)
        antisense_ids = SlidingWindowContig.tokenize_batch(antisense, tokenizer)

        seq_mask = torch.where(seq_ids != tokenizer.pad_token_id, True, False)
        antisense_mask = torch.where(
            antisense_ids != tokenizer.pad_token_id, True, False
        )

        pos = [x["pos"] for x in batch]
        name = [x["name"] for x in batch]

        return {
            "seq": seq,
            "antisense": antisense,
            "pos": pos,
            "seq_ids": seq_ids,
            "antisense_ids": antisense_ids,
            "seq_mask": seq_mask,
            "antisense_mask": antisense_mask,
            "name": name,
        }


class SlidingWindowFasta(LenDataset):
    """
    Given a fasta with a single sequence, generate sliding windows
    of a fixed size and stride.  Additionally provide the start/end
    positions of the windows.
    """

    def __init__(
        self,
        filenames: list[str],
        window_size: int,
        stride: int,
        reverse_complement: bool = False,
    ):
        print("SLIDING WINDOW FASTA INITIALIZING...", file=sys.stderr)
        self.fasta_paths = [Path(filename) for filename in filenames]
        print(self.fasta_paths, file=sys.stderr)

        # TODO do with pyfastx just as with FastqDataset
        sequences, sample_names = self.get_sequences()

        # add reverse complements to dataset along with original sequences
        if reverse_complement:
            print("ADDING REVERSE COMPLEMENTS...", file=sys.stderr)
            sequences += [self.reverse_complement(seq) for seq in sequences]
            sample_names += sample_names
        print("FASTAS LOADED...", file=sys.stderr)

        # combine all windowed sequences, positions, and sample names into flat lists
        # maintaining the order of the sequences, positions, and sample names.
        self.windowed_sequences, self.positions, self.sample_names = (
            self.sliding_windows(sequences, window_size, stride, sample_names)
        )
        print("SLIDING WINDOWS GENERATED...", file=sys.stderr)

    def reverse_complement(self, seq: str) -> str:
        # TODO This is probably slow, but for small data not a big deal
        # When I do this with pyfastx, I can just use `.antisense` on a seq
        complement = {"A": "T", "C": "G", "G": "C", "T": "A"}
        return "".join(complement[base] for base in reversed(seq))

    def get_sequences(self) -> tuple[list[str], list[str]]:
        """
        I'm just gonna load the whole fasta into memory.
        Assume only one sequence per fasta.
        """
        sequences = []
        sample_names = []
        for fasta in self.fasta_paths:
            with open(fasta, "r") as f:
                sample_names.append(fasta.stem)
                seq = []
                for line in f:
                    if line.startswith(">"):
                        continue
                    seq.append(line.strip().upper())
                sequences.append("".join(seq))
        return sequences, sample_names

    def sliding_windows(
        self,
        sequences: list[str],
        window_size: int,
        stride: int,
        sample_names: list[str],
    ):
        windowed_seqs = [
            x
            for seq in sequences
            for x in [
                seq[i : i + window_size]
                for i in range(0, len(seq) - window_size + 1, stride)
            ]
        ]
        positions = [
            x
            for seq in sequences
            for x in [
                (i, i + window_size - 1)
                for i in range(0, len(seq) - window_size + 1, stride)
            ]
        ]
        samples = [
            x
            for sample_name, sequence in zip(sample_names, sequences)
            for x in [
                sample_name for _ in range(0, len(sequence) - window_size + 1, stride)
            ]
        ]
        return windowed_seqs, positions, samples

    @staticmethod
    def tokenize_batch(batch, tokenizer):
        return tokenizer.batch_encode_plus(
            batch,
            return_tensors="pt",
            padding="longest",
        )["input_ids"]

    def __len__(self):
        return len(self.windowed_sequences)

    def __getitem__(self, idx):
        return {
            "seq": self.windowed_sequences[idx],
            "pos": self.positions[idx],
            "sample": self.sample_names[idx],
        }

    @staticmethod
    def collate_fn(batch, tokenizer):
        seq = SlidingWindowFasta.tokenize_batch([x["seq"] for x in batch], tokenizer)
        attention_mask = torch.where(seq != tokenizer.pad_token_id, True, False)
        pos = [x["pos"] for x in batch]
        sample = [x["sample"] for x in batch]

        return {
            "input_ids": seq,
            "pos": pos,
            "sample": sample,
            "attention_mask": attention_mask,
        }
