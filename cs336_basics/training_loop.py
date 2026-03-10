import os
import torch
import numpy as np
import wandb
from .transformer_lm import Transformer_LM
from .adamw import AdamW

# 导入你 utils.py 中写好的模块
from .utils import (
    data_loading, 
    cross_entropy, 
    learning_rate_schedule, 
    gradient_clipping, 
    save_checkpoint
)

# ==========================================
# 1. 集中管理超参数 (Hyperparameters)
# ==========================================
config = {
    "batch_size": 32,
    "context_length": 256,
    "max_iters": 5000,          # 总训练步数
    "eval_interval": 100,       # 每隔多少步评估一次验证集
    "eval_iters": 100,          # 每次评估跑多少个 batch 计算平均 loss
    "save_interval": 1000,      # 每隔多少步保存一次 checkpoint
    
    "d_model": 512,
    "n_heads": 16,
    "layers": 4,
    "vocab_size": 10000,        
    
    "learning_rate": 5e-4,      # lr_max
    "min_lr": 1e-5,             # lr_min
    "warmup_iters": 100,        # T_w
    "weight_decay": 0.01,
    "grad_clip": 1.0,           # 最大 L2 norm
    
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    "out_dir": "/home/ubuntu/cs336_assignments/assignment1-basics/checkpoints"
}

# 初始化 wandb
wandb.init(project="train_transformer_LM", mode="offline", config=config)

# ==========================================
# 2. 数据加载与模型初始化
# ==========================================
os.makedirs(config["out_dir"], exist_ok=True)

# 使用 mmap 内存映射模式加载大规模数据 [cite: 1002-1008]
train_data = np.load('/home/ubuntu/cs336_assignments/assignment1-basics/data/tinystories_train_encoded.npy', mmap_mode='r')
val_data = np.load('/home/ubuntu/cs336_assignments/assignment1-basics/data/tinystories_valid_encoded.npy', mmap_mode='r')

model = Transformer_LM(
    d_model=config["d_model"], 
    num_heads=config["n_heads"], 
    d_ff=4 * config["d_model"], 
    vocab_size=config["vocab_size"], 
    context_length=config["context_length"], 
    num_layers=config["layers"], 
    apply_rope=True, 
    theta=10000.0, 
    device=config["device"]
)

optimizer = AdamW(
    model.parameters(), 
    lr=config["learning_rate"], 
    betas=(0.9, 0.95), 
    weight_decay=config["weight_decay"]
)

wandb.watch(model, log="all", log_freq=100)

# ==========================================
# 3. 验证集评估逻辑
# ==========================================
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(config["eval_iters"])
        for k in range(config["eval_iters"]):
            X, Y = data_loading(data, config["batch_size"], config["context_length"], device=config["device"])
            # X shape: (bs, seq_len) -> model -> (bs, seq_len, vocab_size)
            logits = model(X)
            # Reshape for your cross_entropy implementation [cite: 819-828]
            loss = cross_entropy(logits.reshape(-1, config["vocab_size"]), Y.reshape(-1))
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# ==========================================
# 4. 主训练循环 [cite: 1046-1054]
# ==========================================
for iter_num in range(config["max_iters"]):
    
    # 1. 使用你实现的学习率调度器更新当前步的学习率 [cite: 965-967]
    lr = learning_rate_schedule(
        t=iter_num, 
        lr_max=config["learning_rate"], 
        lr_min=config["min_lr"], 
        T_w=config["warmup_iters"], 
        T_c=config["max_iters"]
    )
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # 2. 定期评估并在 W&B 记录
    if iter_num % config["eval_interval"] == 0 or iter_num == config["max_iters"] - 1:
        losses = estimate_loss()
        print(f"Step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        wandb.log({
            "iter": iter_num,
            "train/loss": losses['train'],
            "val/loss": losses['val'],
            "lr": lr
        })

    # 3. 定期使用你实现的 save_checkpoint 保存模型 [cite: 1022-1025]
    if iter_num > 0 and iter_num % config["save_interval"] == 0:
        ckpt_path = os.path.join(config["out_dir"], f"ckpt_iter_{iter_num}.pt")
        save_checkpoint(model=model, optimizer=optimizer, iteration=iter_num, out=ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

    # 4. 获取数据并前向传播
    inputs, targets = data_loading(train_data, config["batch_size"], config["context_length"], device=config["device"])
    output = model(inputs)
    loss = cross_entropy(output.reshape(-1, config["vocab_size"]), targets.reshape(-1))
    
    wandb.log({"train/batch_loss": loss.item()})
    
    # 5. 反向传播与优化
    optimizer.zero_grad()
    loss.backward()
    
    # 使用你实现的梯度裁剪 [cite: 975-978]
    gradient_clipping(parameters=model.parameters(), max_l2_norm=config["grad_clip"])
    
    optimizer.step()

wandb.finish()