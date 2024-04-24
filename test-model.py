"""
For now just use the huggingface library to load the model and test it.
Later, I might mess around with installing it from the git repo and
using pytorch lightning.
"""
# Load model directly
# import torch
# from standalone_hyenadna import CharacterTokenizer
# from huggingface_wrapper import HyenaDNAPreTrainedModel

# # instantiate pretrained model
# pretrained_model_name = 'hyenadna-medium-160k-seqlen'
# max_length = 160_000

# model = HyenaDNAPreTrainedModel.from_pretrained(
#     './checkpoints',
#     pretrained_model_name,
# )

# max_length = 160_000
# # create tokenizer, no training involved :)
# tokenizer = CharacterTokenizer(
#     characters=['A', 'C', 'G', 'T', 'N'],  # add DNA characters
#     model_max_length=max_length,
# )

# # create a sample
# sequence = 'ACTG' * int(max_length/4)
# tok_seq = tokenizer(sequence)["input_ids"]

# # place on device, convert to tensor
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# tok_seq = torch.LongTensor(tok_seq).unsqueeze(0).to(device)  # unsqueeze for batch dim

# # prep model and forward
# model.to(device)
# model.eval()  # deterministic

# with torch.inference_mode():
#     embeddings = model(tok_seq)

# print(embeddings.shape)
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer, logging
from datasets import Dataset
import torch

# instantiate pretrained model
checkpoint = 'LongSafari/hyenadna-medium-160k-seqlen-hf'
max_length = 1024

# bfloat16 for better speed and reduced memory usage
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, torch_dtype=torch.float32, device_map="auto", trust_remote_code=True)

# Generate some random sequence and labels
# If you're copying this code, replace the sequences and labels
# here with your own data!
sequence = 'ACTG' * int(max_length/4)
sequence = [sequence] * 8  # Create 8 identical samples
tokenized = tokenizer(sequence)["input_ids"]
labels = [0, 1] * 4

# Create a dataset for training
ds = Dataset.from_dict({"input_ids": tokenized, "labels": labels})
ds.set_format("pt")

# Initialize Trainer
# Note that we're using extremely small batch sizes to maximize
# our ability to fit long sequences in memory!
args = {
    "output_dir": "tmp",
    "num_train_epochs": 1,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "gradient_checkpointing": True,
    "learning_rate": 2e-5,
}
training_args = TrainingArguments(**args)

trainer = Trainer(model=model, args=training_args, train_dataset=ds)
result = trainer.train()

print(result)

