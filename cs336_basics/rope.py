import torch
from torch import nn

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        theta: float value for the RoPE
        d_k: query 和 key 向量的维度
        max_seq_len: 允许输入的最大序列长度 
        """
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        k = torch.arange(0, d_k, 2, dtype=torch.float32)
        freq = 1.0 / (theta ** (k / d_k))

        # 预计算位置 i 与 频率的乘积
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        anlgles = torch.outer(positions, freq)

        # 计算 cos 和 sin, 非参数，不参与计算
        self.register_buffer("cos_angles", torch.cos(anlgles), persistent=False)
        self.register_buffer("sin_angles", torch.sin(anlgles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None=None) -> torch.Tensor:
        """
        x: [..., seq_len, d_k]
        token_positions: [..., seq_len]
        """
        if token_positions is None:
            seq_len = x.shape[-2]
            token_positions = torch.arange(seq_len, device=x.device)
        token_positions.unsqueeze(-2)
        # 特征对拆分
        x_reshaped = x.unflatten(-1, (self.d_k // 2, 2))

        # 获取当前位置的 cos 和 sin
        cos = self.cos_angles[token_positions]
        sin = self.sin_angles[token_positions]
        cos = cos.unsqueeze(-1)
        sin = sin.unsqueeze(-1)

        # 应用旋转
        x0, x1 = x_reshaped[..., 0:1], x_reshaped[..., 1:2]
        x0_rot = x0 * cos - x1 * sin
        x1_rot = x0 * sin + x1 * cos

        # 恢复形状
        x_rotated = torch.cat([x0_rot, x1_rot], dim=-1)
        return x_rotated.flatten(-2)

