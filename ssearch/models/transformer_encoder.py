from os.path import basename

import lightning as L
import torch
from peft import IA3Config, PeftModel, get_peft_model
from transformers import AutoModelForMaskedLM, AutoTokenizer


class TransformerEncoder(torch.nn.Module):
    """
    Transformer encoder initialized from a huggingface model.
    """

    def __init__(self, model_version, checkpoint=None):
        """
        init with pretrained foundation model and wrap with new or pretrained PEFT adapter
        """
        super().__init__()
        if checkpoint:
            self.model = PeftModel.from_pretrained(
                model=AutoModelForMaskedLM.from_pretrained(
                    model_version, trust_remote_code=True
                ).base_model,
                model_id=checkpoint,
                adapter_name=basename(checkpoint),
            )
            return

        ia3_config = IA3Config(
            target_modules=["key", "value", "dense"], feedforward_modules=["dense"]
        )
        base_model = AutoModelForMaskedLM.from_pretrained(
            model_version, trust_remote_code=True
        ).base_model
        self.model = get_peft_model(
            base_model, ia3_config, adapter_name="nucleotide-transformer-ia3-ssearch"
        )

    def forward(self, input_ids, attention_mask):
        """
        Get output embeddings
        """
        embeddings = self.model(
            input_ids,
            attention_mask=attention_mask,
            encoder_attention_mask=attention_mask,
            output_hidden_states=False,
        ).last_hidden_state
        attention_mask = attention_mask.unsqueeze(-1)
        return torch.sum(attention_mask * embeddings, dim=1) / torch.sum(
            attention_mask, dim=1
        )
