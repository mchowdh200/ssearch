"""
Contains functions for creating a distributed knn reference genome index.
For each contig in a reference genome, we will create a faiss index comprised
of sequence embeddings generated from a sliding window across the contig.

Then to query the reference index, we will perform a query on each index and
pick the results with the highest scores.

The main use case of these functions is to use them in conjuction with a
snakemake pipeline.  But in the future, I may put together a standalone main
script in this file.
"""

from functools import partial
from pathlib import Path

import faiss
import matplotlib.pyplot as plt
import numpy as np
import pyfastx
from transformers import AutoTokenizer

from ssearch.data_utils.datasets import SlidingWindowContig
from ssearch.inference.build_index import build_index
from ssearch.inference.query_index import query_index
from ssearch.models.transformer_encoder import TransformerEncoder


def filter_contigs(
    reference_fasta: Path | str, keep: set[str], output_fasta: Path | str
):
    """
    Print the fasta entries for the contigs in to file
    """
    fasta = pyfastx.Fasta(
        reference_fasta, build_index=True, key_func=lambda x: x.split()[0]
    )
    with open(output_fasta, "w") as f:
        for seq in fasta:
            if seq.name in keep:
                f.write(seq.raw.strip() + "\n")


def write_genomic_positions(batch: dict, output_path: str | Path):
    """
    Write genomic positions for each batch element to metadata file.
    """
    chroms = batch["name"].split()[0]
    pos = batch["pos"]
    with open(output_path, "a") as f:
        for chrom, (start, end) in zip(chroms, pos):
            f.write(f"{chrom}\t{start}\t{end}\n")


def get_contigs(reference_fasta: Path | str) -> list[str]:
    fasta = pyfastx.Fasta(reference_fasta)
    return [key.split()[0] for key in fasta.keys()]


def flat_l2_index(d: int):
    """
    Why? because faiss is a SWIG wrapper around C++ code, and it doesn't accept kwargs...
    """
    return faiss.IndexFlatL2(d)


def build_knn_reference_index(
    reference_fasta: Path,
    window_size: int,
    stride: int,
    output_dir: Path,
    base_model: str,
    adapter_checkpoint: str,
    embedding_dim: int,
    batch_size: int,
    num_workers_per_gpu: int,
    num_gpus: int,
    use_amp: bool,
):
    contigs = get_contigs(reference_fasta)
    datasets = [
        SlidingWindowContig(
            fasta=reference_fasta,
            contig=contig,
            window_size=window_size,
            stride=stride,
        )
        for contig in contigs
    ]

    for contig, dataset in zip(contigs, datasets):

        output_dir = Path(output_dir) / contig
        output_dir.mkdir(parents=True, exist_ok=True)

        build_index(
            model_factory=TransformerEncoder,
            model_args={
                "base_model": base_model,
                "adapter_checkpoint": adapter_checkpoint,
            },
            index_factory=flat_l2_index,
            index_args={"d": embedding_dim},
            output_dir=output_dir,
            batch_size_per_gpu=batch_size,
            output_shape=(embedding_dim,),
            num_workers_per_gpu=num_workers_per_gpu,
            num_gpus=num_gpus,
            use_amp=use_amp,
            datasets=[dataset],
            collate_fn=partial(
                SlidingWindowContig.collate_fn,
                tokenizer=AutoTokenizer.from_pretrained(base_model),
            ),
            worker_init_fn=SlidingWindowContig.worker_init_fn,
            model_input_keys=["input_ids", "attention_mask"],
            metadata_write_fn=partial(
                write_genomic_positions,
                output_path=output_dir / f"{contig}.metadata",
            ),
        )
