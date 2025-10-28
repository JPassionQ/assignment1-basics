import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float=1e-5, device=None, dtype=None):
        """
        d_model: 模型的隐藏层维度
        eps: 固定位1e-5, 用于数值稳定性
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, dtype=dtype, device=device))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, seq_len, d_model]
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms_x = torch.sqrt(torch.sum(x ** 2, dim=-1) / self.d_model + self.eps).unsqueeze(-1)
        result = x / rms_x * self.weight

        return result.to(in_dtype)