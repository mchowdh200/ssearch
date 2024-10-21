import lightning as L
import torch
import torch.nn.functional as F


def constrastive_loss(
    u: torch.Tensor,
    v: torch.Tensor,
    sim: torch.Tensor,
    similarity_threshold: float,
):
    """
    Modified version of contrastive loss that is geared towards continuous similarity
    instead of binary labels. For similar pairs, the loss is only computed if the
    predicted similarity is less than the actual similarity. For dissimilar pairs,
    the loss is only computed if the predicted similarity is greater than the actual
    similarity.
    u, v: embeddings of the input sequences
    sim: similarity between the pairs
    """
    pred_sim = torch.cosine_similarity(u, v)
    return (
        (sim > similarity_threshold)
        * torch.where(
            pred_sim - sim < 0,
            F.mse_loss(pred_sim, sim, reduction="none"), # penalize more for underestimation
            F.huber_loss(pred_sim, sim, reduction="none"),
        )
        + (sim <= similarity_threshold)
        * torch.where(
            pred_sim - sim > 0, F.mse_loss(pred_sim, sim, reduction="none"), 0
        )
    ).mean()


class SiameseModule(L.LightningModule):

    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float,
        weight_decay: float,
        similarity_threshold: float,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.similarity_threshold = similarity_threshold
        self.save_hyperparameters()

    def forward(self, batch: dict):
        """
        A, B: batches input ids of tokenized sequence pairs
        sim: similarity(A, B)
        """
        A, B, A_mask, B_mask, sim = (
            batch["A"],
            batch["B"],
            batch["A_mask"],
            batch["B_mask"],
            batch["sim"],
        )
        u = self.model(A, attention_mask=A_mask)
        v = self.model(B, attention_mask=B_mask)
        with torch.no_grad():
            pred = torch.cosine_similarity(u, v)
            mse = torch.nn.functional.mse_loss(pred, sim)
        loss = constrastive_loss(u, v, sim, self.similarity_threshold)
        return loss, mse

    def training_step(self, batch, _):
        loss, mse = self.forward(batch)
        self.log("train_loss", loss)
        self.log("train_mse", mse)
        return loss

    def validation_step(self, batch, _):
        loss, mse = self.forward(batch)
        self.log("val_loss", loss, on_epoch=True, sync_dist=True)
        self.log("val_mse", mse, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return optimizer
