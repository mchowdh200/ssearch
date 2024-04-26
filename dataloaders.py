import torch
from torch.utils.data import Dataset
import pyfastx

class FastqDataset(Dataset):
    def __init__(self, fastq_file):
        self.fastq = pyfastx.Fastq(fastq_file)

    def __len__(self):
        return len(self.fastq)

    def __getitem__(self, idx):
        seq = self.fastq[idx].seq
        return seq

def tokenize_batch(sequences, tokenizer):
    return torch.LongTensor(tokenizer(sequences)["input_ids"])
