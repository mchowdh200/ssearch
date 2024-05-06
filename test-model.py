import argparse
from functools import partial

import lightning as L
import pyfastx
import torch
import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from models import SiameseModule
from utils import FaissIndexWriter, SequenceDataset, tokenize_batch


def parse_args():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers()

    ## -----------------------------------------------------
    ## subcommand: make_index
    ## -----------------------------------------------------
    make_index_cmd = subparsers.add_parser(
        "make-index", help="Make a faiss index from a fastq file"
    )
    make_index_cmd.add_argument(
        "-c",
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to the (lightning) model checkpoint",
    )
    make_index_cmd.add_argument(
        "-t",
        "--tokenizer-checkpoint",
        type=str,
        default="LongSafari/hyenadna-medium-160k-seqlen-hf",
        help="Path to the huggingface tokenizer checkpoint",
    )
    make_index_cmd.add_argument(
        "-f",
        "--fastq",
        type=str,
        required=True,
        help="Path to the fastq to index",
    )
    make_index_cmd.add_argument(
        "-i",
        "--index-path",
        type=str,
        required=True,
        help="Path to save the faiss index",
    )
    make_index_cmd.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=4096,
        help="Batch size",
    )
    make_index_cmd.add_argument(
        "-e",
        "--embeddings-dim",
        type=int,
        default=256,
        help="Dimension of the embedding vectors",
    )
    make_index_cmd.add_argument(
        "-d",
        "--devices",
        type=int,
        default=1,
        help="Number of gpus to use",
    )
    make_index_cmd.set_defaults(func=make_index)

    ## -----------------------------------------------------
    ## TODO subcommand: search
    ## -----------------------------------------------------
    # search_cmd = subparsers.add_parser(
    #     "search", help="query the faiss index"
    # )

    return parser.parse_args()


def make_index(args):
    model = SiameseModule.load_from_checkpoint(args.checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_checkpoint, trust_remote_code=True
    )
    seqs = [x.seq for x in pyfastx.Fastq(args.fastq)]
    dataset = SequenceDataset(seqs)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=partial(tokenize_batch, tokenizer=tokenizer),
    )
    index_writer = FaissIndexWriter(args.index_path, args.embeddings_dim)
    trainer = L.Trainer(
        devices=args.devices,
        strategy="auto",
        callbacks=[index_writer],
    )
    trainer.predict(model, dataloaders=[dataloader], return_predictions=False)


def query_index(args):
    pass


if __name__ == "__main__":
    args = parse_args()
    args.func(args)
