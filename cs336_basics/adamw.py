from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(
            self,
            params,
            lr: float,
            betas: tuple[float],
            weight_decay: float,
            eps: float=1e-8,
        ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, beta1=betas[0], beta2=betas[1], weight_decay=weight_decay, eps=eps)
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            beta1 = group['beta1']
            beta2 = group['beta2']
            weight_decay = group['weight_decay']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]  # 获取该参数的状态字典
                if len(state) == 0: # 第一次访问时初始化状态
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    state['v'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

                step = state['step']
                m = state['m']
                v = state['v']
                grad = p.grad.data

                # 状态和参数更新
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad ** 2
                lr_t = lr * (math.sqrt(1-beta2 ** (step+1)) / (1 - beta1 ** (step+1)))
                p.data -= lr_t * (m / (torch.sqrt(v) + eps)) # 参数更新
                p.data -= lr * weight_decay * p.data  # 权重衰减

                state['step'] = step + 1
                state['m'] = m
                state['v'] = v
        return loss