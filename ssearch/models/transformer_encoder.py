import torch
from peft import IA3Config, IA3Model
from transformers import AutoModelForMaskedLM, AutoTokenizer


class TransformerEncoder(torch.nn.Module):
    """
    Transformer encoder initialized from a huggingface model.
    """

    def __init__(self, model_version):
        super().__init__()
        ia3_config = IA3Config()
        self.model = IA3Model(
            AutoModelForMaskedLM.from_pretrained(model_version, trust_remote_code=True),
            ia3_config,
            adapter_name="nucleotide-transformer-ia3-ssearch",
        )

    def forward(self, input_ids, attention_mask):
        """
        Get output embeddings
        """
        embeddings = self.model(
            input_ids,
            attention_mask=attention_mask,
            encoder_attention_mask=attention_mask,
            output_hidden_states=True,
        )["hidden_states"][-1]
        attention_mask = attention_mask.unsqueeze(-1)
        return torch.sum(attention_mask * embeddings, dim=1) / torch.sum(
            attention_mask, dim=1
        )
