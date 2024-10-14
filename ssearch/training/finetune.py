import lightning as L
import torch
from config import DefaultConfig
from data_utils.datasets import SiameseDataset, get_tokenizer
from lightning.pytorch import callbacks
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from models.transformer_encoder import TransformerEncoder


def main():
    train_dataset = SiameseDataset(
        data=DefaultConfig.TrainingData.TRAIN_DATA,
        max_length=DefaultConfig.Model.SEQUENCE_LENGTH,
        base_model=DefaultConfig.Model.BASE_MODEL,
    )
    val_dataset = SiameseDataset(
        data=DefaultConfig.TrainingData.VAL_DATA,
        max_length=DefaultConfig.Model.SEQUENCE_LENGTH,
        base_model=DefaultConfig.Model.BASE_MODEL,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=DefaultConfig.Trainer.BATCH_SIZE,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=DefaultConfig.Trainer.BATCH_SIZE,
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
    )

    model = TransformerEncoder(model_version=DefaultConfig.Model.BASE_MODEL)
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    logger = WandbLogger(
        name=DefaultConfig.Logging.NAME,
        project=DefaultConfig.Logging.PROJECT,
        log_model=False,
    )
    logger.log_hyperparams(DefaultConfig.to_dict())

    trainer_callbacks = [
        callbacks.ModelCheckpoint(
            DefaultConfig.Logging.CHECKPOINT_DIR,
            monitor="val_loss",
            filename="{epoch}-{val_loss:.4f}",
            save_top_k=1,
            mode="min",
        ),
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            mode="min",
        ),
    ]

    trainer = L.Trainer(
        devices=DefaultConfig.Trainer.DEVICES,
        accelerator="auto",
        max_epochs=DefaultConfig.Trainer.EPOCHS,
        logger=logger,
        callbacks=trainer_callbacks,
    )
    trainer.fit(model, train_loader, val_loader)
