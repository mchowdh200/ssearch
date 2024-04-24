"""
For now just use the huggingface library to load the model and test it.
Later, I might mess around with installing it from the git repo and
using pytorch lightning.
"""
# Load model directly
import torch
from standalone_hyenadna import CharacterTokenizer
from huggingface_wrapper import HyenaDNAPreTrainedModel

# instantiate pretrained model
pretrained_model_name = 'hyenadna-medium-160k-seqlen'
max_length = 160_000

model = HyenaDNAPreTrainedModel.from_pretrained(
    './checkpoints',
    pretrained_model_name,
)

max_length = 160_000
# create tokenizer, no training involved :)
tokenizer = CharacterTokenizer(
    characters=['A', 'C', 'G', 'T', 'N'],  # add DNA characters
    model_max_length=max_length,
)

# create a sample
sequence = 'ACTG' * int(max_length/4)
tok_seq = tokenizer(sequence)["input_ids"]

# place on device, convert to tensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok_seq = torch.LongTensor(tok_seq).unsqueeze(0).to(device)  # unsqueeze for batch dim

# prep model and forward
model.to(device)
model.eval()  # deterministic

with torch.inference_mode():
    embeddings = model(tok_seq)

print(embeddings.shape)
