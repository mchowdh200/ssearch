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
from transformers.models.audio_spectrogram_transformer.feature_extraction_audio_spectrogram_transformer import \
    window_function

from models import SiameseModule
from utils import (FaissIndexWriter, FaissQueryWriter,
                   SlidingWindowReferenceFasta, tokenize_batch)


def build_reference_index(args):
    model = SiameseModule.load_from_checkpoint(args.checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_checkpoint, trust_remote=True
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
    build_cmd.add_argument(
        "--num-workers",
        type=int,
        help="Number of workers for the dataloader",
    )
    build_cmd.set_defaults(func=build_reference_index)

    return parser.parse_args()
