import argparse
import os
import sys
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional

import faiss
from transformers import AutoTokenizer

from ssearch.config import DefaultConfig
from ssearch.data_utils.datasets import (FastqDataset, LenDataset,
                                         SlidingWindowFasta)
from ssearch.inference.build_index import build_index
from ssearch.inference.query_index import query_index
from ssearch.models.transformer_encoder import TransformerEncoder


def not_implemented(**kwargs):
    raise NotImplementedError("TODO")


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
    plot_parser = subparsers.add_parser("plot")

    ## build index args --------------------------------------------------------
    build_parser.add_argument(
        "--fastqs",
        type=str,
        nargs="+",
        default=DefaultConfig.Inference.METAGENOMIC_INDEX_DATA,
        help="Paths to fastq files to index.",
    )
    build_parser.set_defaults(func=build_metagenomics_index)
    # TODO add more options about the index type, etc.

    ## search index args -------------------------------------------------------
    search_parser.add_argument(
        "--query-fastas",
        type=str,
        nargs="+",
        default=DefaultConfig.Inference.METAGENOMIC_QUERY_DATA,
        help="Path to fastq file to search against index.",
    )
    search_parser.set_defaults(func=query_metagenomics_index)

    ## plot args ---------------------------------------------------------------
    # TODO add args
    plot_parser.set_defaults(func=not_implemented)

    ## common args -------------------------------------------------------------
    for name, subp in subparsers.choices.items():
        subp.add_argument(
            "--output-dir",
            type=str,
            default=DefaultConfig.Inference.OUTPUT_DIR,
        )
        if name != "plot":
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


def build_metagenomics_index(
    fastqs: list[str],
    output_dir: str,
    base_model: str,
    adapter_checkpoint: str,
    batch_size: int,
    num_workers_per_gpu: int,
    num_gpus: int,
    use_amp: bool,
):

    datasets: list[LenDataset] = [
        FastqDataset(fastq, remake_index=False) for fastq in fastqs
    ]
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

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
        datasets=datasets,
        collate_fn=partial(
            FastqDataset.basic_collate_fn, tokenizer=tokenizer, upper_case=True
        ),
        worker_init_fn=FastqDataset.worker_init_fn,
    )


# TODO make some of this configurable with DefaultConfig
def query_metagenomics_index(
    query_fastas: list[str],
    output_dir: str,
    base_model: str,
    adapter_checkpoint: str,
    batch_size: int,
    num_workers_per_gpu: int,
    num_gpus: int,
    use_amp: bool,
):
    dataset = SlidingWindowFasta(query_fastas, window_size=150, stride=50)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    query_index(
        model_factory=TransformerEncoder,
        model_args={"model_version": base_model, "checkpoint": adapter_checkpoint},
        index_path="TODO",
        output_dir=output_dir,
        batch_size_per_gpu=batch_size,
        num_workers_per_gpu=num_workers_per_gpu,
        num_gpus=num_gpus,
        use_amp=use_amp,
        datasets=[dataset],
        collate_fn=partial(
            SlidingWindowFasta.collate_fn,
            tokenizer=tokenizer,
        ),
        model_input_keys=["input_ids", "attention_mask"],
        metadata_write_fn=write_sample_pos,
        k=10,
    )


if __name__ == "__main__":
    args = parse_args()

    func = args.func
    kwargs = vars(args)
    del kwargs["func"]

    func(**kwargs)
