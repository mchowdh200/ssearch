import argparse
import itertools
from functools import partial

import faiss
import numpy as np
import pyfastx
import torch
from torch.utils.data import DataLoader

from utils import SlidingWindowFasta, load_model
from os.path import splitext, basename


def query(index, model, seqs, topk, device):
    with torch.no_grad():
        embeddings = torch.mean(model(seqs.to(device)).last_hidden_state, dim=1)
    return index.search(embeddings.cpu().numpy(), topk)


def main(
    index_path: str,
    model_checkpoint: str,
    batch_size: int,
    query_fastas: list[str],
    window_size: int,
    stride: int,
    topk: int,
    output: str,
):
    index = faiss.read_index(index_path)
    model, tokenizer = load_model(model_checkpoint)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(output, "w") as f:
        for fasta in query_fastas:
            sliding_window_fasta = SlidingWindowFasta(
                fasta, window_size=window_size, stride=stride
            )
            sample_name = splitext(basename(fasta))[0]
            dataloader = DataLoader(
                sliding_window_fasta,
                batch_size=batch_size,
                collate_fn=partial(SlidingWindowFasta.collate_fn, tokenizer=tokenizer),
            )

            for batch in dataloader:
                seqs = batch["seq"]
                pos = batch["pos"]
                # D: shape (batch_size, topk)
                D, _ = query(index, model, seqs, topk, device)

                for d, p in zip(D, pos):
                    score = np.mean(d)
                    f.write(f"{sample_name}\t{p[0]}\t{p[1]}\t{score}\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index-path",
        type=str,
        required=True,
        help="Path to the faiss index",
    )
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default="LongSafari/hyenadna-medium-160k-seqlen-hf",
        help="Huggingface model checkpoint",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--query-fastas",
        type=str,
        nargs="+",
        required=True,
        help="Paths to the query fasta file space separated",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=150,
        help="Break the query sequence into windowed queries of this size",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=50,
        help="Stride of the sliding window",
    )
    parser.add_argument(
        "-k",
        "--topk",
        type=int,
        default=10,
        help="Number of top results to return in knn query",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output file",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
