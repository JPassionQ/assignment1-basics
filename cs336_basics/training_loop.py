import torch
from .transformer_lm import Transformer_LM
from .tokenizer import Tokenizer
from .utils import data_loading, cross_entropy
from .adamw import AdamW
import numpy as np
from tests.test_tokenizer import get_tokenizer_from_vocab_merges_path
import wandb

# ==========================================
# 1. 集中管理超参数
# ==========================================
config = {
    "batch_size": 16,
    "context_length": 256,
    "epochs": 1000,
    "d_model": 512,
    "n_heads": 16,
    "layers": 4,
    "vocab_size": 50257,
    "lr": 5e-4,
    "weight_decay": 0.01,
    "max_seq_length": 1024,
    "dataset": "tinystories_train"
}

# 初始化 wandb，设置离线模式
wandb.init(
    project="train_transformer_LM", 
    mode="offline",
    config=config
)

encoded_text = np.load('/home/ubuntu/cs336_assignments/assignment1-basics/data/tinystories_train_encoded.npy')
encoded_text = encoded_text.astype(np.int64)

# 使用 config 初始化模型
model = Transformer_LM(
    d_model=config["d_model"], 
    num_heads=config["n_heads"], 
    d_ff=4 * config["d_model"], 
    vocab_size=config["vocab_size"], 
    context_length=config["max_seq_length"], 
    num_layers=config["layers"], 
    apply_rope=True, 
    theta=10000.0, 
    device="cuda:0"
)

optimizer = AdamW(
    model.parameters(), 
    lr=config["lr"], 
    betas=(0.9, 0.999), 
    weight_decay=config["weight_decay"]
)

# 监听模型权重和梯度
wandb.watch(model, log="all", log_freq=100)

for i in range(config["epochs"]):
    optimizer.zero_grad()
    inputs, targets = data_loading(encoded_text, config["batch_size"], config["context_length"], device="cuda:0")
    
    output = model(inputs)
    output_flat = output.reshape(-1, config["vocab_size"])
    targets_flat = targets.reshape(-1)
    
    loss = cross_entropy(output_flat, targets_flat)
    
    print(f"epoch: {i}, loss: {loss.item()}")
    
    # ==========================================
    # 2. 核心：只需在这里记录，W&B 会自动画图
    # ==========================================
    wandb.log({
        "train_loss": loss.item(),
        "epoch": i
    })
    
    loss.backward()
    optimizer.step()

# ==========================================
# 3. 优雅结束
# ==========================================
wandb.finish()