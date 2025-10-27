import torch
from torch import nn
from einops import einsum, rearrange
import math
from jaxtyping import Bool, Float, Int
from torch import Tensor

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
