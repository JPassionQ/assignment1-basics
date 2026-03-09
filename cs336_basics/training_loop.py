import torch
from .transformer_lm import Transformer_LM
from .tokenizer import Tokenizer
from .utils import data_loading, cross_entropy
from .adamw import AdamW
import numpy as np
from tests.test_tokenizer import get_tokenizer_from_vocab_merges_path
import matplotlib.pyplot as plt

corpus_file = "/home/jingqi/CS336_Assignments/assignment1-basics/tests/fixtures/tinystories_sample_5M.txt"
vocab_file = "/home/jingqi/CS336_Assignments/assignment1-basics/tests/fixtures/gpt2_vocab.json"
merges_file = "/home/jingqi/CS336_Assignments/assignment1-basics/tests/fixtures/gpt2_merges.txt"

tokenizer = get_tokenizer_from_vocab_merges_path(vocab_file, merges_file, special_tokens=['<unk>', '<pad>', '<sos>', '<eos>'])

with open(corpus_file, "r", encoding="utf-8") as f:
    text = f.read()

encoded_text = np.array(tokenizer.encode(text), dtype=np.int64)

batch_size = 8
context_length = 128

epoch = 1000
d_model= 768
n_heads = 12
layers = 12
vocab_size = 50257
lr = 5e-4
weight_decay=0.01
max_seq_length = 1024
model = Transformer_LM(d_model=d_model, num_heads=n_heads, d_ff=4 * d_model, vocab_size=vocab_size, context_length=max_seq_length, num_layers=layers, apply_rope=True, theta=10000.0, device="cuda:0")
optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)

# Track losses for plotting
losses = []

for i in range(epoch):
    optimizer.zero_grad()
    inputs, targets = data_loading(encoded_text, batch_size, context_length, device="cuda:0")
    output = model(inputs)
    output_flat = output.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)
    loss = cross_entropy(output_flat, targets_flat)
    
    # Store loss value
    losses.append(loss.item())
    
    print(f"epoch: {i}, loss: {loss}")
    loss.backward()
    optimizer.step()

# Plot loss vs epoch
plt.figure(figsize=(10, 6))
plt.plot(range(epoch), losses, 'b-', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training Loss vs Epoch', fontsize=14)
plt.grid(True, alpha=0.3)
plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
plt.show()
print("Loss plot saved as 'training_loss.png'")