import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def get_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


class SiameseDataset(Dataset):
    """
    Load tab separated file with columns A, B, sim
    where A, B are sequences and sim is a float similarity score.
    """

    def __init__(self, data: str, base_model: str):
        # We're assuming the data fits in memory.
        # Otherwise we'd need to use some kind of
        # lazy loading alternative to the dataframe.
        self.data = pd.read_csv(data, sep="\t", names=["A", "B", "sim"])
        self.tokenizer = get_tokenizer(base_model)
        self.base_model = base_model

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx]

    def tokenize_batch(self, batch: list[str]):
        return self.tokenizer.batch_encode_plus(
            batch,
            return_tensors="pt",
            padding="longest",
        )["input_ids"]

    def collate_fn(self, batch: list[dict]):
        A, B, sim = zip(*[(x["A"], x["B"], x["sim"]) for x in batch])

        return {
            "A": self.tokenize_batch(A),
            "B": self.tokenize_batch(B),
            "sim": torch.FloatTensor(sim),
        }
