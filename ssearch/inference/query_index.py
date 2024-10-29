import os
from pathlib import Path
from typing import Any, Callable, Optional

import faiss
import numpy as np
import torch

from ssearch.data_utils.datasets import LenDataset
from ssearch.inference.distributed_inference import DistributedInference
from numpy.lib.format import open_memmap


def query_index(
    model_factory: Callable[..., torch.nn.Module],
    model_args: dict,
    index_path: str,
    output_dir: str,
    batch_size_per_gpu: int,
    num_workers_per_gpu: int,
    num_gpus: int,
    use_amp: bool,
    datasets: list[LenDataset],
    collate_fn: Callable[[Any], Any],
    worker_init_fn: Optional[Callable[[Any], None]] = None,
    model_input_keys: Optional[list[str]] = None,
    metadata_write_fn: Optional[Callable[..., None]] = None,
    k: Optional[int] = None,
):
    os.makedirs(output_dir, exist_ok=True)

    for i, dataset in enumerate(datasets):
        DistributedInference(
            model_factory=model_factory,
            model_args=model_args,
            dataset=dataset,
            output_path=Path(f"{output_dir}/query_dataset_{i}.npy"),
            batch_size=batch_size_per_gpu,
            dataloader_num_workers=num_workers_per_gpu,
            dataloader_worker_init_fn=worker_init_fn,
            dataloader_collate_fn=collate_fn,
            model_input_keys=model_input_keys,
            metadata_write_fn=metadata_write_fn,
            num_gpus=num_gpus,
            use_amp=use_amp,
        ).run()

    index = faiss.read_index(index_path)
    mmaps = [
        np.load(f"{output_dir}/query_embeddings_{i}.npy", mmap_mode="r")
        for i in range(len(datasets))
    ]

    D_out = open_memmap(
        f"{output_dir}/query_results_D.npy",
        mode="w+",
        shape=(sum(len(dataset) for dataset in datasets), k),
        dtype=np.int64,
    )
    I_out = open_memmap(
        f"{output_dir}/query_results_I.npy",
        mode="w+",
        shape=(sum(len(dataset) for dataset in datasets), k),
        dtype=np.int64,
    ) 

    for mmap in mmaps:
        for i in range(0, len(mmap), batch_size_per_gpu):
            D, I = index.search(mmap[i : i + batch_size_per_gpu], k)
            D_out[i : i + batch_size_per_gpu] = D
            I_out[i : i + batch_size_per_gpu] = I

    D_out.flush()
    I_out.flush()

