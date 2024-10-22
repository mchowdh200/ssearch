import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from ssearch.config import DefaultConfig
from ssearch.data_utils.datasets import SiameseDataset, get_tokenizer
from ssearch.models.callbacks import PEFTAdapterCheckpoint
from ssearch.models.siamese import SiameseModule
from ssearch.models.transformer_encoder import TransformerEncoder


def main():
    train_dataset = SiameseDataset(
        data=DefaultConfig.TrainingData.TRAIN_DATA,
        # max_length=DefaultConfig.Model.SEQUENCE_LENGTH,
        base_model=DefaultConfig.Model.BASE_MODEL,
    )
    val_dataset = SiameseDataset(
        data=DefaultConfig.TrainingData.VAL_DATA,
        # max_length=DefaultConfig.Model.SEQUENCE_LENGTH,
        base_model=DefaultConfig.Model.BASE_MODEL,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=DefaultConfig.Trainer.BATCH_SIZE,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=DefaultConfig.Trainer.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=DefaultConfig.Trainer.BATCH_SIZE,
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
        num_workers=DefaultConfig.Trainer.NUM_WORKERS,
    )

    torch.set_float32_matmul_precision("high")

    model = TransformerEncoder(model_version=DefaultConfig.Model.BASE_MODEL)
    model = model.to(
        torch.device(torch.device("cuda") if torch.cuda.is_available() else "cpu")
    )
    siamese_module = SiameseModule(
        model,
        learning_rate=DefaultConfig.Trainer.LEARNING_RATE,
        weight_decay=DefaultConfig.Trainer.WEIGHT_DECAY,
        similarity_threshold=DefaultConfig.Trainer.SIMILARITY_THRESHOLD,
    )

    logger = WandbLogger(
        name=DefaultConfig.Logging.NAME,
        project=DefaultConfig.Logging.PROJECT,
        log_model=False,
    )
    logger.log_hyperparams(DefaultConfig.to_dict())

    trainer_callbacks = [
        PEFTAdapterCheckpoint(
            DefaultConfig.Logging.CHECKPOINT_DIR,
            monitor="val_loss",
            filename="{epoch}-{val_loss:.4f}",
            save_top_k=1,
            mode="min",
            module_name="model.model"
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            mode="min",
        ),
    ]

    trainer = L.Trainer(
        devices=DefaultConfig.Trainer.DEVICES,
        accelerator="auto",
        precision="16-mixed",
        strategy=(
            "ddp_find_unused_parameters_true"
            if DefaultConfig.Trainer.DEVICES > 1
            else "auto"
        ),
        accumulate_grad_batches=DefaultConfig.Trainer.ACCUMULATE_GRAD_BATCHES,
        max_epochs=DefaultConfig.Trainer.EPOCHS,
        callbacks=trainer_callbacks,
        logger=logger,
    )
    trainer.fit(siamese_module, train_loader, val_loader)
