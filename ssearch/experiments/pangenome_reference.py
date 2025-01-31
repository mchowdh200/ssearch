from functools import partial
from pathlib import Path

import faiss
# import matplotlib.pyplot as plt
# import numpy as np
import pyfastx
from transformers import AutoTokenizer

from ssearch.data_utils.datasets import SlidingWindowContig
from ssearch.inference.build_index import build_index
from ssearch.inference.query_index import query_index
from ssearch.models.transformer_encoder import TransformerEncoder


def write_genomic_positions(batch: dict, output_path: str | Path):
    """
    Write genomic positions for each batch element to metadata file.
    """
    chroms = [name.split()[0] for name in batch["name"]]
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


def build_pangenome_reference_index(
    reference_fasta: Path,
    window_size: int,
    stride: int,
    output_dir: Path,
    base_model: str,
    adapter_checkpoint: Path,
    embedding_dim: int,
    batch_size: int,
    num_workers_per_gpu: int,
    num_gpus: int,
    use_amp: bool,
):
    datasets = [
        SlidingWindowFastx(  # TODO
            fasta=reference_fasta,
            orientation=orientation,  # TODO
            window_size=window_size,
            stride=stride,
        )
        for orientation in ["sense", "antisense"]
    ]

    build_index(
        model_factory=TransformerEncoder,
        model_args={
            "model_version": base_model,
            "checkpoint": adapter_checkpoint,
        },
        index_factory=flat_l2_index,
        index_args={"d": embedding_dim},
        output_dir=output_dir,
        batch_size_per_gpu=batch_size,
        output_shape=(embedding_dim,),
        num_workers_per_gpu=num_workers_per_gpu,
        num_gpus=num_gpus,
        use_amp=use_amp,
        datasets=datasets,
        collate_fn=partial(
            SlidingWindowContig.collate_fn,
            tokenizer=AutoTokenizer.from_pretrained(base_model),
        ),
        worker_init_fn=SlidingWindowContig.worker_init_fn,
        model_input_keys=["input_ids", "attention_mask"],
        metadata_write_fn=partial(
            write_genomic_positions,
            output_path=output_dir / "index_ids.bed",
        ),
    )
