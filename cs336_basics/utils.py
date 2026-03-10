import torch
from torch import nn
from einops import einsum, rearrange
import math
from jaxtyping import Bool, Float, Int
from torch import Tensor
from collections.abc import Iterable
import numpy as np
import numpy.typing as npt
from typing import IO, Any, BinaryIO
import os


def softmax(x: torch.Tensor, dim: int):
    # 减去最大值，保持数值稳定性
    max_values = torch.max(x, dim=dim, keepdim=True).values
    x_shifted = x - max_values

    exp_x = torch.exp(x_shifted)
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)

    result = exp_x / sum_exp

    return result

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None):
    """
    缩放点积注意力实现
    Q: (bs, ..., seq_len, d_k)
    K: (bs, ..., seq_len, d_k)
    V: (bs, ..., seq_len, d_v)
    mask: (seq_len, seq_len)

    return: (bs, ..., d_v)
    """
    d_k = Q.shape[-1]
    attn_score = einsum(Q, K, "... i d_k, ... j d_k -> ... i j") / math.sqrt(d_k)
    if mask is not None:
        attn_score = attn_score.masked_fill(~mask, float('-inf'))
    attn_weights = softmax(x=attn_score, dim=-1)
    result = einsum(attn_weights, V, "... i j, ... j d_v -> ... i d_v")
    return result

def cross_entropy(
        inputs: Float[Tensor, " batch_size vocab_size"], 
        targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    max_values = torch.max(inputs, dim=-1, keepdim=True).values.detach()
    inputs_shifted = inputs - max_values

    negative_targets_logits_sum = torch.sum(-inputs_shifted[torch.arange(inputs.shape[0]), targets])

    exp_inputs = torch.exp(inputs_shifted)
    exp_sum = torch.sum(exp_inputs, dim=-1, keepdim=True)
    log_sum = torch.sum(torch.log(exp_sum))


    bs = inputs.shape[0]
    loss = (negative_targets_logits_sum + log_sum) / bs
    return loss

def learning_rate_schedule(
        t: int,
        lr_max: float,
        lr_min: float,
        T_w: int,
        T_c: int
):
    """
    有 warmup 阶段的 余弦学习率 调度
    """
    # warm-up 阶段
    if t < T_w:
        return t / T_w * lr_max
    # 余弦退火
    if T_w <= t <= T_c:
        return lr_min + 0.5 * (1 + math.cos((t - T_w)/(T_c - T_w) * math.pi)) * (lr_max - lr_min)
    # post-annealing
    if t > T_c:
        return lr_min

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    total_norm = 0
    param_list = list(parameters)
    for param in param_list:
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2

    total_norm = math.sqrt(total_norm)
    # 增加 1e-6 防止分母为0 
    clip_coef = max_l2_norm / (total_norm + 1e-6) 
    
    if clip_coef < 1:
        for param in param_list:  # <--- 修复：使用 param_list 而不是 parameters
            if param.grad is not None:
                param.grad.data.mul_(clip_coef)
    
    return total_norm


def data_loading(x: npt.NDArray,
                batch_size: int,
                context_length: int,
                device: str) -> tuple[torch.Tensor, torch.Tensor]:
    max_start_idx = len(x) - context_length - 1
    if max_start_idx <= 0:
        raise ValueError("数据集长度小于所需的最小长度")

    # 随机采样 bs 个起始索引，不包含右端点，所以+1
    start_indices = np.random.randint(0, max_start_idx + 1, size=batch_size)
    
    # 初始化输入和目标数组
    inputs = np.empty((batch_size, context_length), dtype=x.dtype)
    targets = np.empty((batch_size, context_length), dtype=x.dtype)

    # 填充输入和目标序列
    for i, start in enumerate(start_indices):
        end = start + context_length
        inputs[i] = x[start: end]
        targets[i] = x[start + 1: end + 1]
    
    # 转换为torch张量并转移到指定设备
    inputs_tensor = torch.tensor(inputs, dtype=torch.long, device=device)
    targets_tensor = torch.tensor(targets, dtype=torch.long, device=device)

    return inputs_tensor, targets_tensor

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(checkpoint, out)

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]