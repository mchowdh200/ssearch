import pyfastx
import torch
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer


def load_model(checkpoint: str):
    model = AutoModel.from_pretrained(
        checkpoint, device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    return model, tokenizer


class SlidingWindowFasta(Dataset):
    """
    Given a fasta with a single sequence, generate sliding windows
    of a fixed size and stride.  Additionally provide the start/end
    positions of the windows.
    """

    def __init__(self, fasta_path, window_size, stride):
        fasta = pyfastx.Fasta(fasta_path)
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
        return {"seq": self.windowed_sequences[idx], "pos": self.positions[idx]}

    @staticmethod
    def collate_fn(batch: list[dict], tokenizer):
        return {
            "seq": tokenize_batch([x["seq"] for x in batch], tokenizer),
            "pos": [x["pos"] for x in batch],
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


def tokenize_batch(batch: list[str], tokenizer):
    return torch.LongTensor(tokenizer(batch)["input_ids"])
