import os
import sys
from pathlib import Path
from typing import Any, Callable, Optional

import faiss
import numpy as np
import torch

from ssearch.data_utils.datasets import LenDataset
from ssearch.inference.distributed_inference import DistributedInference


def build_index(
    model_factory: Callable[..., torch.nn.Module],
    model_args: dict,
    index_factory: Callable[..., faiss.Index],
    index_args: dict,
    output_dir: str | Path,
    batch_size_per_gpu: int,
    output_shape: tuple,
    num_workers_per_gpu: int,
    num_gpus: int,
    use_amp: bool,
    datasets: list[LenDataset],
    collate_fn: Callable[[Any], Any],
    worker_init_fn: Optional[Callable[[Any], None]] = None,
    model_input_keys: Optional[list[str]] = None,
    metadata_write_fn: Optional[Callable[..., None]] = None,
):
    """
    Load fine tuned model, and run distributed inference for each dataset, and
    save result mmaps to output_dir.
    """

    os.makedirs(output_dir, exist_ok=True)

    # Run distributed inference for each dataset
    for i, dataset in enumerate(datasets):
        print(f"Running inference for dataset {i}/{len(datasets)}", file=sys.stderr)
        DistributedInference(
            model_factory=model_factory,
            model_args=model_args,
            dataset=dataset,
            output_path=Path(f"{output_dir}/dataset_{i}.npy"),
            batch_size=batch_size_per_gpu,
            output_shape=output_shape,
            dataloader_num_workers=num_workers_per_gpu,
            dataloader_worker_init_fn=worker_init_fn,
            dataloader_collate_fn=collate_fn,
            model_input_keys=model_input_keys,
            metadata_write_fn=metadata_write_fn,
            num_gpus=num_gpus,
            use_amp=use_amp,
        ).run()

    print("Initializing index", file=sys.stderr)
    index = index_factory(**index_args)

    print("Loading mmaps", file=sys.stderr)
    mmaps = [
        np.load(f"{output_dir}/dataset_{i}.npy", mmap_mode="r")
        for i in range(len(datasets))
    ]

    # TODO make config param
    print("Building index", file=sys.stderr)
    index_batch_size = 256
    for mmap in mmaps:
        for i in range(0, len(mmap), index_batch_size):
            index.add(mmap[i : i + index_batch_size])

    print("Saving index", file=sys.stderr)
    faiss.write_index(index, f"{output_dir}/index.faiss")
