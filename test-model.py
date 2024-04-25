"""
For now just use the huggingface library to load the model and test it.
Later, I might mess around with installing it from the git repo and
using pytorch lightning.
"""
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
from transformers import TrainingArguments, Trainer, logging
from datasets import Dataset
import torch

# instantiate pretrained model
checkpoint = 'LongSafari/hyenadna-medium-160k-seqlen-hf'
max_length = 1024 # whatever you want it to be upto the model's true max len

# bfloat16 for better speed and reduced memory usage
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModel.from_pretrained(checkpoint, device_map="auto", trust_remote_code=True)

# Generate some random sequence and labels
# If you're copying this code, replace the sequences and labels
# here with your own data!
device = 'cuda'
sequence = 'ACTG' * int(max_length/4)
sequence = [sequence] * 8  # Create 8 identical samples
tokenized = torch.LongTensor(tokenizer(sequence)["input_ids"]).to(device)

# aggregate output over the sequence dimension
result = torch.mean(model(tokenized).last_hidden_state, dim=1)
print(result.shape)
