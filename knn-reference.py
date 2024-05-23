docstring = """
Module to build/test a vector KNN reference Genome.
"""
import argparse
import os
from functools import partial

import lightning as L
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from models import SiameseModule
from utils import (FaissIndexWriter, FastqDataset, KNNReferenceQueryWriter,
                   SlidingWindowReferenceFasta, tokenize_batch)


def make_strided_bed(args):
    """
    Make a strided bed file from a reference fasta file for future dataloader use.
    """
    # init function makes a bed file from a reference
    # fasta file and saves it to the fasta's directory.
    dataset = SlidingWindowReferenceFasta(
        fasta_path=args.fasta,
        window_size=args.window_size,
        stride=args.stride,
        chromosome_names=args.chromosome_names,
    )


def build_reference_index(args):
    """
    Build a knn reference genome from a reference fasta file
    """
    model = SiameseModule.load_from_checkpoint(args.checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_checkpoint, trust_remote_code=True
    )

    dataset = SlidingWindowReferenceFasta(
        fasta_path=args.fasta,
        window_size=args.window_size,
        stride=args.stride,
        chromosome_names=args.chromosome_names,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=partial(SlidingWindowReferenceFasta.collate_fn, tokenizer=tokenizer),
    )

    index_writer = FaissIndexWriter(
        index_path=dataset.fasta_path + ".faiss", dim=args.embeddings_dim, with_ids=True
    )

    trainer = L.Trainer(
        devices=args.devices,
        strategy="auto",
        callbacks=[index_writer],
    )
    trainer.predict(model, dataloaders=[dataloader], return_predictions=False)


def query_reference_index(args):
    """
    Query a knn reference genome and write the results to a file with format
    filename    seq_name1    chrm:start-end,dist1    chrm:start-end,dist2    ...
    filename    seq_name2    ...
    """
    model = SiameseModule.load_from_checkpoint(args.checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_checkpoint, trust_remote_code=True
    )

    datasets = [FastqDataset(filename=fq) for fq in args.query_fastqs]
    dataloaders = [
        DataLoader(
            dset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            collate_fn=partial(dset.collate_fn, tokenizer=tokenizer),
        )
        for dset in datasets
    ]

    query_writer = KNNReferenceQueryWriter(
        index=args.faiss_index,
        id_bed=args.windowed_bed,
        topk=args.topk,
        output=args.output,
    )
    trainer = L.Trainer(
        devices=args.devices,
        strategy="auto",
        callbacks=[query_writer],
    )
    trainer.predict(model, dataloaders=dataloaders, return_predictions=False)


def parse_args():
    parser = argparse.ArgumentParser(description=docstring)
    subparsers = parser.add_subparsers()

    ## -----------------------------------------------------------------------
    ## Build knn-reference genome
    ## -----------------------------------------------------------------------
    build_cmd = subparsers.add_parser(
        "build", help="Build a knn reference genome from reference fasta"
    )
    build_cmd.add_argument("--fasta", help="Path to reference fasta file", type=str)
    build_cmd.add_argument(
        "--chromosome-names",
        type=str,
        nargs="+",
        help="Chromosome names to include. Defaults to hg38 style names",
        default=["chr" + str(i) for i in range(1, 23)] + ["chrX", "chrY", "chrM"],
    )
    build_cmd.add_argument(
        "--window-size",
        type=int,
        help="Size of the window to use for the reference genome",
        default=150,
    )
    build_cmd.add_argument(
        "--stride",
        type=int,
        help="Stride of the sliding window",
        default=50,
    )
    build_cmd.add_argument(
        "--outdir",
        type=str,
        help="Path to write the output files",
        default=os.getcwd(),
    )
    build_cmd.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to the (lightning) model checkpoint",
    )
    build_cmd.add_argument(
        "--tokenizer-checkpoint",
        type=str,
        default="LongSafari/hyenadna-medium-160k-seqlen-hf",
        help="Path to the huggingface tokenizer checkpoint",
    )
    build_cmd.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Batch size for dataloader",
    )
    build_cmd.add_argument(
        "--embeddings-dim",
        type=int,
        default=256,
        help="Dimension of the embedding vectors",
    )
    build_cmd.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of gpus to use",
    )
    build_cmd.add_argument(
        "--num-workers",
        type=int,
        help="Number of workers for the dataloader",
    )
    build_cmd.set_defaults(func=build_reference_index)

    ## -----------------------------------------------------------------------
    ## Make strided bed file from reference
    ## -----------------------------------------------------------------------
    make_bed_cmd = subparsers.add_parser(
        "make-bed", help="Make a bed file from a reference fasta"
    )
    make_bed_cmd.add_argument(
        "--fasta",
        help="Path to reference fasta file",
        type=str,
    )
    make_bed_cmd.add_argument(
        "--chromosome-names",
        type=str,
        nargs="+",
        help="Chromosome names to include. Defaults to hg38 style names",
        default=["chr" + str(i) for i in range(1, 23)] + ["chrX", "chrY", "chrM"],
    )
    make_bed_cmd.add_argument(
        "--window-size",
        type=int,
        help="Size of the window to use for the reference genome",
        default=150,
    )
    make_bed_cmd.add_argument(
        "--stride",
        type=int,
        help="Stride of the sliding window",
        default=50,
    )
    make_bed_cmd.set_defaults(func=make_strided_bed)

    ## -----------------------------------------------------------------------
    ## query knn-reference genome
    ## -----------------------------------------------------------------------
    query_cmd = subparsers.add_parser("query", help="Query a knn reference genome")
    query_cmd.add_argument(
        "--faiss-index",
        help="Path to the faiss reference index file",
        type=str,
    )
    query_cmd.add_argument(
        "--windowed-bed",
        help="Path to the windowed bed file",
        type=str,
    )
    query_cmd.add_argument(
        "--query-fastqs",
        nargs="+",
        help="Paths to the query fastq files",
        type=str,
    )
    query_cmd.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to the (lightning) model checkpoint",
    )
    query_cmd.add_argument(
        "--tokenizer-checkpoint",
        type=str,
        default="LongSafari/hyenadna-medium-160k-seqlen-hf",
        help="Path to the huggingface tokenizer checkpoint",
    )
    query_cmd.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Batch size for dataloader",
    )
    query_cmd.add_argument(
        "--devices",
        type=int,
        required=True,
        help="Number of gpus to use",
    )
    query_cmd.add_argument(
        "--num-workers",
        type=int,
        required=True,
        help="Number of workers for the dataloader",
    )
    query_cmd.add_argument(
        "--topk",
        type=int,
        required=True,
        help="Number of top hits to return",
    )
    query_cmd.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to write the output file",
    )
    query_cmd.set_defaults(func=query_reference_index)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.func(args)
