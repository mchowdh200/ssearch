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
from utils import FaissIndexWriter, FaissQueryWriter, tokenize_batch


def get_chromosome_lengths(fasta_index: str, chromosome_names: set) -> dict[str, int]:
    """
    Read fai to get the chromosome lengths.
    """
    with open(fasta_index) as f:
        return {
            x[0]: int(line.split("\t")[1])
            for line in f
            if (x := line.split("\t"))[0] in chromosome_names
        }


def make_strided_bed(
    outfile: str, chroms: dict[str, int], window_size: int, stride: int
):
    """
    Generate a bed file for a set of chromosomes with a sliding window.
    - outfile: path to write the bed intervals
    - chroms: dict of chromosome lengths
    - window_size: size of the window
    - stride: stride of the sliding window
    """
    with open(outfile, "w") as f:
        for chrom, length in chroms.items():
            # write bed intervals along with a unique int id
            for i, start in enumerate(range(0, length - window_size + 1, stride)):
                end = start + window_size
                f.write(f"{chrom}\t{start}\t{end}\t{i}\n")


def build_reference_index(args):
    model = SiameseModule.load_from_checkpoint(args.checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_checkpoint, trust_remote=True
    )
    # TODO make a dataloader for the reference genome


def parse_args():
    parser = argparse.ArgumentParser(description=docstring)
    subparsers = parser.add_subparsers()
    build_cmd = subparsers.add_parser(
        "build", help="Build a knn reference genome from reference fasta"
    )
    build_cmd.add_argument("--fasta", help="Path to reference fasta file", type=str)
    build_cmd.add_argument(
        "--fasta-index",
        help="Path to reference fasta index file",
        type=str,
    )
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
    build_cmd.set_defaults(func=build_reference_index)

    return parser.parse_args()
