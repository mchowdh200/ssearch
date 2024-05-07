import argparse
from functools import partial

import lightning as L
import pyfastx
import torch
import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from models import SiameseModule
from plot_results import make_plot
from utils import (FaissIndexWriter, FaissQueryWriter, SequenceDataset,
                   SlidingWindowFasta, tokenize_batch)


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
    search_cmd = subparsers.add_parser("search", help="query the faiss index")
    search_cmd.add_argument(
        "-i",
        "--index-path",
        type=str,
        required=True,
        help="Path to the faiss index",
    )
    search_cmd.add_argument(
        "-c",
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to the (lightning) model checkpoint",
    )
    search_cmd.add_argument(
        "-t",
        "--tokenizer-checkpoint",
        type=str,
        default="LongSafari/hyenadna-medium-160k-seqlen-hf",
        help="Path to the huggingface tokenizer checkpoint",
    )
    search_cmd.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=4096,
        help="Batch size",
    )
    search_cmd.add_argument(
        "-f",
        "--fastas",
        type=str,
        nargs="+",
        required=True,
        help="Paths to the query fastas",
    )
    search_cmd.add_argument(
        "-w",
        "--window-size",
        type=int,
        default=150,
        help="Break the query sequence into windowed queries of this size",
    )
    search_cmd.add_argument(
        "-s",
        "--stride",
        type=int,
        default=50,
        help="Stride of the windowed queries across the fasta sequences",
    )
    search_cmd.add_argument(
        "-k",
        "--topk",
        type=int,
        default=10,
        help="Number of top hits to return",
    )
    search_cmd.add_argument(
        "-d",
        "--devices",
        type=int,
        default=1,
        help="Number of gpus to use",
    )
    search_cmd.add_argument(
        "-q",
        "--query-output",
        type=str,
        default="scored-queries.bed",
        help="Path to save the search results",
    )
    search_cmd.add_argument(
        "-r",
        "--results-output",
        type=str,
        default="aggregated-results.png",
        help="Path to save the search results plot",
    )
    search_cmd.set_defaults(func=query_index)

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
    ## Load model, datasets, etc. --------------------------------------------
    model = SiameseModule.load_from_checkpoint(args.checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_checkpoint, trust_remote_code=True
    )
    query_writer = FaissQueryWriter(
        index_path=args.index_path,
        topk=args.topk,
        output=args.query_output,
        clear_existing=True,
    )

    datasets = [
        SlidingWindowFasta(fasta, args.window_size, args.stride)
        for fasta in args.fastas
    ]
    dataloaders = [
        DataLoader(
            dataset,
            batch_size=args.batch_size,
            collate_fn=partial(SlidingWindowFasta.collate_fn, tokenizer=tokenizer),
        )
        for dataset in datasets
    ]

    ## Run the search --------------------------------------------------------
    trainer = L.Trainer(
        devices=args.devices,
        strategy="auto",
        callbacks=[query_writer],
    )
    trainer.predict(model, dataloaders=dataloaders, return_predictions=False)

    ## Aggregate the results -------------------------------------------------
    make_plot(args.query_output, args.results_output)


if __name__ == "__main__":
    args = parse_args()
    args.func(args)
