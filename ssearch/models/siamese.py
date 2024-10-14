import torch
import lightning as L

class SiameseModule(L.LightningModule):

    def __init__(self, model: torch.nn.Module, learning_rate: float, weight_decay: float):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.save_hyperparameters()

    def forward(self, batch: dict):
        """
        A, B: batches input ids of tokenized sequence pairs
        sim: similarity(A, B)
        """
        A, B, sim = batch["A"], batch["B"], batch["sim"]
        u = self.model(A)
        v = self.model(B)
        pred = torch.cosine_similarity(u, v)
        return torch.nn.functional.mse_loss(pred, sim)

    def training_step(self, batch, _):
        loss = self.forward(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, _):
        loss = self.forward(batch)
        self.log("val_loss", loss, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return optimizer
