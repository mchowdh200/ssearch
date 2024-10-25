from abc import ABC, abstractmethod
from os.path import exists

import pandas as pd
import pyfastx
import torch
from torch.utils.data import Dataset
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


class FastqDataset(LenDataset):
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
        collate_fn=dataset.collate_fn,
    )
    """

    def __init__(
        self,
        filename: str,
        # base_model: str,
        # upper_case: bool,
        remake_index=False,
    ):
        self.filename = filename
        if not exists(filename + ".fxi") or remake_index:
            self.make_index()

        self.tokenizer = get_tokenizer(base_model)
        self.upper_case = upper_case

    def make_index(self):
        """build the index before initializing workers."""
        pyfastx.Fastq(self.filename, build_index=True)

    @staticmethod
    def tokenize_batch(batch, tokenizer, upper_case):
        return tokenizer.batch_encode_plus(
            [b.upper() for b in batch] if upper_case else batch,
            return_tensors="pt",
            padding="longest",
        )["input_ids"]

    def __len__(self):
        fastq = pyfastx.Fastq(self.filename)
        return len(fastq)

    def __getitem__(self, idx):
        return self.fastq[idx]

    @staticmethod
    def basic_collate_fn(batch, tokenizer, upper_case):
        """
        Collate function that returns tokenized sequences and attention masks
        """
        input_ids = FastqDataset.tokenize_batch(
            [x.seq for x in batch], tokenizer, upper_case
        )
        attention_mask = torch.where(input_ids != tokenizer.pad_token_id, True, False)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def metadata_collate_fn(self, batch):
        """
        Collate function that returns metadata along with the tokenized sequences.
        """
        raise NotImplementedError()
        # return {
        #     "filename": [self.filename for _ in batch],
        #     "seq": self.tokenize_batch([x.seq for x in batch], tokenizer),
        #     "name": [x.name for x in batch],
        #     "qual": [x.qual for x in batch],
        # }

    @staticmethod
    def worker_init_fn(worker_id):
        worker_info = get_worker_info()
        dataset = worker_info.dataset
        fastq = pyfastx.Fastq(dataset.filename)
        dataset.fastq = fastq
