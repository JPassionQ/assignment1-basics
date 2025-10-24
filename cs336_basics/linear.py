import torch
from torch import nn

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None=None, dtype:torch.dtype | None=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
 
        self.W = nn.Parameter(torch.empty(out_features, in_features ,device=device, dtype=dtype))

        std = torch.math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(self.W, mean=0, std=std, a = -3*std, b = 3 * std)


    def forward(self, x:torch.Tensor) -> torch.Tensor:
        output = x.matmul(self.W.T)
        return output