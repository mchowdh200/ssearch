## Small scale version of the knn reference experiment.
# TODO instead of making this a python script, make it a snakemake pipeline
# That way, I can dispatch the inference jobs to the gpu-cluster and the other jobs to the cpu-cluster?

import argparse
import os
from functools import partial
from pathlib import Path

import faiss
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

from ssearch.config import DefaultConfig
from ssearch.data_utils.datasets import LenDataset, SlidingWindowFasta
from ssearch.inference.build_index import build_index
from ssearch.inference.query_index import query_index
from ssearch.models.transformer_encoder import TransformerEncoder


def write_sample_pos(batch: dict, output_path: str | Path):
    samples = batch["sample"]
    positions = batch["pos"]
    with open(output_path, "a") as f:
        for sample, (start, end) in zip(samples, positions):
            f.write(f"{sample}\t{start}\t{end}\n")


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    build_parser = subparsers.add_parser("build-index")
    search_parser = subparsers.add_parser("search-index")
    plot_parser = subparsers.add_parser("plot-results")

    ## build index args -------------------------------------------------------
    build_parser.add_argument(
        "--reference-fasta",
        type=str,
        default=DefaultConfig.KNNReference.REFERENCE_FASTA,
        help="Input reference fasta file.",
    )

    build_parser.set_defaults(func=build_knn_ref)

    ## search index args ------------------------------------------------------
    search_parser.add_argument(
        "--query-fastas",
        type=str,
        nargs="+",
        help="Input query fasta files.",
    )
    search_parser.add_argument(
        "--k",
        type=int,
        help="Number of nearest neighbors to search.",
        default=DefaultConfig.KNNReference.K,
    )
    search_parser.set_defaults(func=search_knn_ref)

    ## common args -------------------------------------------------------------
    for name, subp in subparsers.choices.items():
        subp.add_argument(
            "--output-dir",
            type=str,
            default=DefaultConfig.Inference.OUTPUT_DIR,
        )
        if name != "plot":
            subp.add_argument(
                "--window-size",
                type=int,
                default=DefaultConfig.KNNReference.WINDOW_SIZE,
                help="Sliding window kmers size.",
            )
            subp.add_argument(
                "--stride",
                type=int,
                default=DefaultConfig.KNNReference.STRIDE,
                help="Sliding window stride.",
            )
            subp.add_argument(
                "--base-model",
                type=str,
                default=DefaultConfig.Inference.BASE_MODEL,
                help="Base huggingface model string.",
            )
            subp.add_argument(
                "--adapter-checkpoint",
                type=str,
                default=DefaultConfig.Inference.ADAPTER_CHECKPOINT,
                help="Path to fine-tuned adapter model checkpoint.",
            )
            subp.add_argument(
                "--batch-size",
                type=int,
                default=DefaultConfig.Inference.BATCH_SIZE,
                help="Model batch size for inference.",
            )
            subp.add_argument(
                "--num-workers-per-gpu",
                type=int,
                default=DefaultConfig.Inference.NUM_WORKERS_PER_GPU,
            )
            subp.add_argument(
                "--num-gpus",
                type=int,
                default=DefaultConfig.Inference.NUM_GPUS,
            )
            subp.add_argument(
                "--use-amp",
                type=bool,
                default=DefaultConfig.Inference.USE_AMP,
            )
    return parser.parse_args()


def flat_l2_index(d: int):
    """
    Why? because faiss is a SWIG wrapper around C++ code, and it doesn't accept kwargs...
    """
    return faiss.IndexFlatL2(d)


def build_knn_ref(
    reference_fasta: str,
    window_size: int,
    stride: int,
    output_dir: str,
    base_model: str,
    adapter_checkpoint: str,
    batch_size: int,
    num_workers_per_gpu: int,
    num_gpus: int,
    use_amp: bool,
):
    # Load reference fasta
    dataset = SlidingWindowFasta(
        filenames=[reference_fasta],
        window_size=window_size,
        stride=stride,
        reverse_complement=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Build index
    build_index(
        model_factory=TransformerEncoder,
        model_args={"model_version": base_model, "checkpoint": adapter_checkpoint},
        index_factory=flat_l2_index,
        index_args={"d": 512},
        output_dir=output_dir,
        batch_size_per_gpu=batch_size,
        output_shape=(512,),
        num_workers_per_gpu=num_workers_per_gpu,
        num_gpus=num_gpus,
        use_amp=use_amp,
        datasets=[dataset],
        collate_fn=partial(SlidingWindowFasta.collate_fn, tokenizer=tokenizer),
        model_input_keys=["input_ids", "attention_mask"],
        metadata_write_fn=write_sample_pos,
    )


def search_knn_ref():
    raise NotImplementedError()

if __name__ == "__main__":
    args = parse_args()

    func = args.func
    kwargs = vars(args)
    del kwargs["func"]

    func(**kwargs)
