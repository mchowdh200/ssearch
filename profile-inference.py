import argparse

import lightning as L
from transformers import AutoTokenizer

from models import SiameseModule
from utils import FastqDataModule


def profile_fastq_dataloader(
    model_checkpoint: str,
    tokenizer_checkpoint: str,
    query_fastqs: list[str],
    num_workers: int,
    batch_size: int,
    devices: int,
):
    model = SiameseModule.load_from_checkpoint(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_checkpoint, trust_remote_code=True
    )

    datamodule = FastqDataModule(
        filenames=query_fastqs,
        tokenizer=tokenizer,
        num_workers=num_workers,  # needs to be > 0 so we can use worker_init_fn
        batch_size=batch_size,
    )

    trainer = L.Trainer(
        devices=devices,
        strategy="auto",
        profiler="simple",
    )

    trainer.predict(model, datamodule, return_predictions=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Profile the inference pipline")
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        required=True,
        help="Path to the lightning module checkpoint",
    )
    parser.add_argument(
        "--tokenizer_checkpoint",
        type=str,
        default="LongSafari/hyenadna-medium-160k-seqlen-hf",
        help="huggingface tokenizer checkpoint",
    )
    parser.add_argument(
        "--query_fastqs",
        type=str,
        nargs="+",
        required=True,
        help="Paths to the query fastq files",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        required=True,
        help="Number of workers for dataloader",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="Batch size for dataloader and inference",
    )
    parser.add_argument(
        "--devices", type=int, required=True, help="Number of gpus to use"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    profile_fastq_dataloader(**vars(args))
