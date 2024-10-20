import lightning as L
import torch


def compute_margin(sim: torch.Tensor, similarity_threshold: float, margin_max: float):
    """
    Margin for constrastive loss that depends on the similarity between examples
    """
    return (margin_max / 2) * (
        1 + torch.cos((2 * torch.pi / similarity_threshold) * sim)
    )


def constrastive_loss(
    u: torch.Tensor,
    v: torch.Tensor,
    sim: torch.Tensor,
    similarity_threshold: float,
    margin_max: float,
):
    """
    Version of contrastive loss that is geared towards continuous similarity instead of binary labels.
    u, v: embeddings of the input sequences
    sim: similarity between the pairs
    """
    margin = compute_margin(sim, similarity_threshold, margin_max)
    pred_sim = torch.cosine_similarity(u, v)
    error = torch.functional.norm(pred_sim - sim, p=2)
    diff = margin - error

    return torch.where(diff > 0, error, 0)


class SiameseModule(L.LightningModule):

    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float,
        weight_decay: float,
        similarity_threshold: float,
        margin_max: float,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.similarity_threshold = similarity_threshold
        self.margin_max = margin_max
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
        pred = torch.cosine_similarity(u, v)
        mse = torch.nn.functional.mse_loss(pred, sim)
        loss = constrastive_loss(u, v, sim, self.similarity_threshold, self.margin_max)
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
