import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


class TransformerEncoder(torch.nn.Module):
    """
    Transformer encoder initialized from a huggingface model.
    """

    def __init__(self, model_version):
        super().__init__()
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_version, trust_remote_code=True
        )
        # we're just loading the tokenizer to get the pad token id
        tokenizer = AutoTokenizer.from_pretrained(model_version, trust_remote_code=True)
        self.pad_token_id = tokenizer.pad_token_id

    def forward(self, input_ids):
        """
        Get output embeddings
        """
        attention_mask = input_ids != self.pad_token_id
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
