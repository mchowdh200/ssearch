import torch
from torch.utils.data import Dataset
import pyfastx

class SequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

def tokenize_batch(batch: list[str], tokenizer):
    return torch.LongTensor(tokenizer(batch)["input_ids"])
