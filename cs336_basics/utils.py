import torch
from torch import nn
from einops import einsum, rearrange

def softmax(x: torch.Tensor, index: int):
    # 减去最大值，保持数值稳定性
    max_values = torch.max(x, dim=index, keepdim=True).values
    x_shifted = x - max_values

    exp_x = torch.exp(x_shifted)
    sum_exp = torch.sum(exp_x, dim=index, keepdim=True)

    result = exp_x / sum_exp

    return result
