import argparse

import lightning as L
import torch
from lightning.pytorch import callbacks
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from models import SiameseModule
from utils import SiameseDataset, load_model

# def train_dataloader(self):
#     dataset = SiameseDataset(self.train_data, self.tokenizer)
#     return DataLoader(
#         dataset,
#         batch_size=self.batch_size,
#         shuffle=True,
#         collate_fn=dataset.collate_fn,
#         pin_memory=True,
#         num_workers=self.num_workers,
#     )

# def val_dataloader(self):
#     dataset = SiameseDataset(self.val_data, self.tokenizer)
#     return DataLoader(
#         dataset,
#         batch_size=self.batch_size,
#         collate_fn=dataset.collate_fn,
#         num_workers=self.num_workers,
#         pin_memory=True,
#         shuffle=False,
#     )


def main(args):
    model, tokenizer = load_model(args.pretrained_checkpoint)
    model = SiameseModule(
        model, learning_rate=args.learning_rate, weight_decay=args.weight_decay
    )

    train_dataset = SiameseDataset(args.train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    val_dataset = SiameseDataset(args.val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=val_dataset.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
    )

    logger = WandbLogger(
        name=args.name,
        project=args.project,
        log_model=False,
    )

    trainer_callbacks = [
        callbacks.ModelCheckpoint(
            args.checkpoint_dir,
            monitor="val_loss",
            filename="best",
            save_top_k=5,
            mode="min",
        ),
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            mode="min",
        ),
        callbacks.RichProgressBar() if args.progress_bar else None,
    ]

    trainer = L.Trainer(
        devices=args.devices,
        accelerator="auto",
        max_epochs=args.epochs,
        logger=logger,
        callbacks=trainer_callbacks,
    )

    trainer.fit(model, train_loader, val_loader)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument(
        "--pretrained-checkpoint",
        type=str,
        default="LongSafari/hyenadna-medium-160k-seqlen-hf",
        help="Pretrained model checkpoint from Hugging Face.",
    )
    parser.add_argument(
        "--train-data",
        type=str,
        required=True,
        help="Training data text file.",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        required=True,
        help="Validation data text file.",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of GPUs to use.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs to train.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="Batch size for dataloaders.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers for dataloaders.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for optimizer.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0,
        help="Weight decay for optimizer.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="fine-tune",
        help="Name of the run.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="seq-similarity",
        help="Wandb project name.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
