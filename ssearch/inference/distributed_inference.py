import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, TypeVar

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from numpy.lib.format import open_memmap
from numpy.typing import DTypeLike
from torch.amp import autocast
from torch.utils.data import DataLoader, Sampler, default_collate

from ssearch.data_utils.datasets import LenDataset

T_co = TypeVar("T_co", covariant=True)


class DistributedInferenceSampler(Sampler[T_co]):
    """
    Distributed sampler geared towards inference. Modified to not drop
    or pad data in the event of uneven batches across replicas.
    """

    def __init__(
        self,
        dataset: LenDataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = len(self.dataset) // self.num_replicas
        if rank < len(self.dataset) % self.num_replicas:
            self.num_samples += 1
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self) -> Iterator[T_co]:
        indices = list(range(len(self.dataset)))

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


def setup_process(rank: int, world_size: int):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    print(f"Rank {rank} initializing process group", file=sys.stderr)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )


@dataclass(kw_only=True)
class DistributedInference:
    # model args
    model_factory: Callable[..., torch.nn.Module]
    model_args: dict

    # dataset args
    dataset: LenDataset
    output_path: Path
    batch_size: int

    # dataloader args
    dataloader_num_workers: int
    dataloader_worker_init_fn: Optional[Callable[[Any], None]] = None
    dataloader_collate_fn: Callable[[Any], Any] = default_collate
    model_input_keys: Optional[list[str]] = None  # for dict inputs
    metadata_write_fn: Optional[Callable[..., None]] = None

    # memmap args
    output_shape: Optional[tuple] = None
    dtype: DTypeLike = np.float32
    
    # inference args
    num_gpus: int = 1
    use_amp: bool = True
    amp_dtype: torch.dtype = torch.float16

    def __post_init__(self):
        # Create output directory if it doesn't exist
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def worker(self, rank: int, world_size: int):
        """Worker process function."""
        setup_process(rank, world_size)

        ## setup model and dataloader -----------------------------------------
        print(f"Rank {rank} initializing model", file=sys.stderr)
        model = self.model_factory(**self.model_args).to(rank)
        model.eval()

        # Set up distributed sampler and dataloader
        print(f"Rank {rank} initializing dataloader", file=sys.stderr)
        sampler = DistributedInferenceSampler(
            self.dataset, num_replicas=world_size, rank=rank
        )

        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.dataloader_num_workers,
            pin_memory=True,
            collate_fn=self.dataloader_collate_fn,
            worker_init_fn=self.dataloader_worker_init_fn,
        )

        ## setup mmap ---------------------------------------------------------
        print(f"Rank {rank} initializing memmap", file=sys.stderr)
        samples_per_worker = sampler.num_samples

        mmap_shape = (
            (samples_per_worker,) + self.output_shape
            if self.output_shape
            else (samples_per_worker,)
        )
        mmap_path = (
            self.output_path.parent
            / f"{self.output_path.stem}_rank{rank}{self.output_path.suffix}"
        )
        # mmap = np.memmap(mmap_path, dtype=self.dtype, mode="w+", shape=mmap_shape)
        mmap = open_memmap(mmap_path, dtype=self.dtype, mode="w+", shape=mmap_shape)

        ## Run inference ------------------------------------------------------
        print(f"Rank {rank} starting inference", file=sys.stderr)
        torch.backends.cudnn.benchmark = True

        with torch.no_grad():
            start_idx = 0
            for batch in dataloader:
                if self.metadata_write_fn:
                    self.metadata_write_fn(batch, output_path=f"{mmap_path}.metadata")
                if isinstance(batch, (tuple, list)):
                    inputs = batch[0].to(rank, non_blocking=True)
                elif isinstance(batch, dict):
                    if self.model_input_keys:
                        inputs = {
                            k: v.to(rank, non_blocking=True)
                            for k, v in batch.items()
                            if k in self.model_input_keys
                        }
                    else:
                        inputs = {
                            k: v.to(rank, non_blocking=True) for k, v in batch.items()
                        }
                else:
                    inputs = batch.to(rank, non_blocking=True)

                if self.use_amp:
                    with autocast(device_type="cuda", dtype=self.amp_dtype):
                        if isinstance(inputs, dict):
                            outputs = model(**inputs)
                        else:
                            outputs = model(inputs)
                else:
                    if isinstance(inputs, dict):
                        outputs = model(**inputs)
                    else:
                        outputs = model(inputs)

                # TODO if batch size is not constant, need to handle this
                batch_outputs = outputs.cpu().numpy()
                batch_size = batch_outputs.shape[0]
                mmap[start_idx : start_idx + batch_size] = batch_outputs
                start_idx += batch_size

                if start_idx % 1000 == 0:
                    print(
                        f"Rank {rank}: {start_idx}/{samples_per_worker}",
                        file=sys.stderr,
                    )
                    mmap.flush()
                    if self.use_amp:
                        torch.cuda.empty_cache()

        mmap.flush()
        dist.barrier()
        dist.destroy_process_group()

    def run(self):
        """Launch distributed inference across all available GPUs."""
        if self.use_amp:
            if self.amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                print(
                    "Warning: bfloat16 not supported on this GPU. Falling back to float16."
                )
                self.amp_dtype = torch.float16

        world_size = max(self.num_gpus, torch.cuda.device_count())
        mp.spawn(self.worker, args=(world_size,), nprocs=world_size, join=True)

        # Combine results from all workers
        all_outputs = []
        for rank in range(world_size):
            mmap_path = (
                self.output_path.parent
                / f"{self.output_path.stem}_rank{rank}{self.output_path.suffix}"
            )
            # mmap = np.memmap(mmap_path, dtype=self.dtype, mode="r")
            mmap = np.load(mmap_path, mmap_mode="r")
            all_outputs.append(mmap)

        # Create final combined memmap
        total_samples = sum(output.shape[0] for output in all_outputs)
        final_shape = (
            (total_samples,) + self.output_shape
            if self.output_shape
            else (total_samples,)
        )
        final_mmap = open_memmap(
            self.output_path, dtype=self.dtype, mode="w+", shape=final_shape
        )

        # Copy data from worker memmaps to final memmap
        start_idx = 0
        for output in all_outputs:
            # for i in range(0, output.shape[0], self.batch_size * 4):
            #     j = min(i + self.batch_size * 4, output.shape[0])
            #     final_mmap[start_idx + i : start_idx + j] = output[i:j]
            size = output.shape[0]
            final_mmap[start_idx : start_idx + size] = output
            start_idx += size

        # Cleanup temporary files
        final_mmap.flush()
        for rank in range(world_size):
            mmap_path = (
                self.output_path.parent
                / f"{self.output_path.stem}_rank{rank}{self.output_path.suffix}"
            )
            os.remove(mmap_path)
