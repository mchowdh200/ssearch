import argparse

import tqdm
import faiss
import numpy as np
import pyfastx
import torch
from functools import partial
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer
from datasets import Dataset
from pprint import pprint

from dataloaders import SequenceDataset, tokenize_batch


def load_model(checkpoint: str):
    model = AutoModel.from_pretrained(
        checkpoint, device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    return model, tokenizer


def main(checkpoint, index_path, dim, fastq, batch_size):
    ## prepare model and data ----------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(checkpoint)
    model.eval()

    fastq = [x.seq for x in pyfastx.Fastq(fastq)]
    dataset = SequenceDataset(fastq)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True,
        collate_fn=partial(tokenize_batch, tokenizer=tokenizer),
    )

    ## build index ---------------------------------------------------------
    index = faiss.IndexFlatL2(dim)
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader):
            embeddings = torch.mean(model(batch.to(device)).last_hidden_state, dim=1)
            index.add(embeddings.cpu().numpy())

    faiss.write_index(index, index_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="LongSafari/hyenadna-medium-160k-seqlen-hf",
        help="checkpoint to load the model and tokenizer",
    )
    parser.add_argument(
        "--index_path",
        type=str,
        default="index.faiss",
        help="path to save the index",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=256,
        help="dimension of the embedding vectors",
    )
    parser.add_argument(
        "--fastq",
        type=str,
        required=True,
        help="fastq file to index",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="batch size for indexing",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
