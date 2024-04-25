"""
For now just use the huggingface library to load the model and test it.
Later, I might mess around with installing it from the git repo and
using pytorch lightning.
"""
import sys

import Levenshtein
import numpy as np
import pyfastx
import torch
from datasets import Dataset
from transformers import AutoModel, AutoTokenizer


def compare_sequences(a, b, max_length=1024):
    """
    Compare two sequences using the Needleman-Wunsch algorithm.
    Truncate sequnces to max_length before comparision.
    """
    if len(a) > max_length:
        a = a[:max_length]
    if len(b) > max_length:
        b = b[:max_length]
    return Levenshtein.distance(a, b)


def compare_embeddings(A: list[str], B: list[str], max_length=1024) -> np.ndarray:
    """
    Compare sequence embeddings using euclidean distance.
    A, B: batches of a, b sequence pairs

    """
    print(type(A))
    print(type(A[0]))
    print(type(B))
    print(type(B[0]))

    # truncate sequences to max_length
    for i in range(len(A)):
        if len(A[i]) > max_length:
            A[i] = A[i][:max_length]
        if len(B[i]) > max_length:
            B[i] = B[i][:max_length]

    # instantiate pretrained model
    checkpoint = "LongSafari/hyenadna-medium-160k-seqlen-hf"

    # bfloat16 for better speed and reduced memory usage
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        checkpoint, device_map="auto", trust_remote_code=True
    )

    # Generate some random sequence and labels
    # If you're copying this code, replace the sequences and labels
    # here with your own data!
    device = "cuda" if torch.cuda.is_available() else "cpu"
    A_tokenized = torch.LongTensor(tokenizer(A)["input_ids"]).to(device)
    B_tokenized = torch.LongTensor(tokenizer(B)["input_ids"]).to(device)

    # aggregate output over the sequence dimension
    A_emb = torch.mean(model(A_tokenized).last_hidden_state, dim=1)
    B_emb = torch.mean(model(B_tokenized).last_hidden_state, dim=1)
    return torch.nn.functional.pairwise_distance(A_emb, B_emb).cpu().detach().numpy()


def rank_embeddings(sequences: list[str], query: str, max_length=1024):
    A = [query] * len(sequences)
    B = sequences

    D = compare_embeddings(A, B, max_length)

    distances = {}
    for i, seq in enumerate(sequences):
        distances[i] = (seq, D[i])

    ranked_distances = dict(sorted(distances.items(), key=lambda x: x[1][1]))
    return ranked_distances




def rank_sequences(sequences: list[str], query: str, max_length=1024):
    """
    Rank sequences by similarity to query using Levenshtein distance.
    """

    # compute pairwise distances between query and sequences
    distances = {}
    for i, seq in enumerate(sequences):
        distance = compare_sequences(seq, query, max_length)
        distances[i] = (seq, distance)

    # sort sequences by distance
    ranked_distances = dict(sorted(distances.items(), key=lambda x: x[1][1]))
    return ranked_distances


if __name__ == "__main__":
    # Load a few sequences from a fasta file
    # Take the first as the query and the rest as the sequences
    # Then do pairwise disance comparison and ranking
    fasta = sys.argv[1]
    fasta = pyfastx.Fasta(fasta, build_index=True)
    query = fasta[0].seq
    sequences = []
    for i in range(10):
        sequences.append(fasta[i + 1].seq)

    ranked_distances = rank_sequences(sequences, query)
    for ranking, distance in ranked_distances.items():
        # print(f"{ranking}: {distance[0]} - {distance[1]}")
        print(f"{ranking}: {distance[1]}")

    ranked_distances = rank_embeddings(sequences, query)
    print("Embedding distances:")
    for ranking, distance in ranked_distances.items():
        # print(f"{ranking}: {distance[0]} - {distance[1]}")
        print(f"{ranking}: {distance[1]}")
