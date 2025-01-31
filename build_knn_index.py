"""
This is a temporary script to build the knn index because
I will be running this from a container and I don't want to
install ssearch on the container while its still in development.
"""

import argparse
import os
import shutil
from pathlib import Path

from ssearch.experiments.knn_reference import build_knn_reference_index


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-fasta", type=str, required=True)
    parser.add_argument("--window-size", type=int, required=True)
    parser.add_argument("--stride", type=int, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, required=True)
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--adapter-checkpoint", type=str, required=True)
    parser.add_argument("--embedding-dim", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-workers-per-gpu", type=int, required=True)
    parser.add_argument("--num-gpus", type=int, required=True)
    parser.add_argument("--use-amp", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)
    shutil.copy(args.reference_fasta, args.cache_dir)
    build_knn_reference_index(
        reference_fasta=args.reference_fasta,
        window_size=args.window_size,
        stride=args.stride,
        output_dir=Path(args.cache_dir),
        base_model=args.base_model,
        adapter_checkpoint=args.adapter_checkpoint,
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        num_workers_per_gpu=args.num_workers_per_gpu,
        num_gpus=args.num_gpus,
        use_amp=args.use_amp,
    )
    shutil.move(args.cache_dir, f"{args.outdir}/knn_indices")
