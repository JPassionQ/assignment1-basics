import torch
from torch import nn

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = nn.Parameter(torch.empty(d_ff, d_model, dtype=dtype, device=device))
        self.w2 = nn.Parameter(torch.empty(d_model, d_ff, dtype=dtype, device=device))
        self.w3 = nn.Parameter(torch.empty(d_ff, d_model, dtype=dtype, device=device))

        std = torch.math.sqrt(2 / (d_model + d_ff))
        nn.init.trunc_normal_(self.w1, mean=0, std=std, a=-3 * std, b = 3 * std)
        nn.init.trunc_normal_(self.w2, mean=0, std=std, a=-3 * std, b = 3 * std)
        nn.init.trunc_normal_(self.w3, mean=0, std=std, a=-3 * std, b = 3 * std)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_x = x.matmul(self.w1.T)
        silu = (w1_x * torch.sigmoid(w1_x))
        w3_x = x.matmul(self.w3.T)
        return silu * (w3_x) @ self.w2.T