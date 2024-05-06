import lightning as L
import torch
from transformers import AutoModel


class SiameseModule(L.LightningModule):
    """
    Load the huggingface model and provide siamese training loop
    """

    def __init__(
        self,
        checkpoint,
        learning_rate,
        weight_decay,
    ):
        super().__init__()
        self.checkpoint = checkpoint
        self.model = AutoModel.from_pretrained(
            checkpoint,
            # device_map="auto",
            trust_remote_code=True,
        )
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.save_hyperparameters()

    def forward(self, x):
        return torch.mean(self.model(x).last_hidden_state, dim=1)

    def siamese_step(self, batch: dict):
        A, B, sim = batch["A"], batch["B"], batch["sim"]
        u = self.forward(A)
        v = self.forward(B)
        pred = torch.cosine_similarity(u, v)
        return torch.nn.functional.mse_loss(pred, sim)

    def training_step(self, batch, batch_idx):
        loss = self.siamese_step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.siamese_step(batch)
        self.log("val_loss", loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return optimizer
