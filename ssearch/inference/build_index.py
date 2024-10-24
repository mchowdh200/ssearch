import os
import sys
from pathlib import Path
from typing import Any, Callable, Optional

import faiss
import numpy as np
import torch

from ssearch.data_utils.datasets import LenDataset
from ssearch.inference.distributed_inference import DistributedInference


def main(
    model_factory: Callable[..., torch.nn.Module],
    model_args: dict,
    index_factory: Callable[..., faiss.Index],
    index_args: dict,
    output_dir: str,
    batch_size_per_gpu: int,
    num_workers_per_gpu: int,
    num_gpus: int,
    use_amp: bool,
    datasets: list[LenDataset],
    collate_fn: Callable[[Any], Any],
    worker_init_fn: Optional[Callable[[Any], None]] = None,
):
    """
    Load fine tuned model, and run distributed inference for each dataset, and
    save result mmaps to output_dir.
    """

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Run distributed inference for each dataset
    for i, dataset in enumerate(datasets):
        print(f"Running inference for dataset {i}/{len(datasets)}", file=sys.stderr)
        DistributedInference(
            model_factory=model_factory,
            model_args=model_args,
            dataset=dataset,
            output_path=Path(f"{output_dir}/dataset_{i}.mmap"),
            batch_size=batch_size_per_gpu,
            dataloader_num_workers=num_workers_per_gpu,
            dataloader_worker_init_fn=worker_init_fn,
            dataloader_collate_fn=collate_fn,
            num_gpus=num_gpus,
            use_amp=use_amp,
        ).run()

    index = index_factory(**index_args)

    # load mmaps (not all at once)
    mmaps = [
        np.memmap(f"{output_dir}/dataset_{i}.mmap", dtype=np.float32, mode="r")
        for i in range(len(datasets))
    ]

    # TODO make config param
    index_batch_size = 256
    for mmap in mmaps:
        for i in range(0, len(mmap), index_batch_size):
            index.add(mmap[i : i + index_batch_size])

    faiss.write_index(index, f"{output_dir}/index.faiss")


