from __future__ import annotations

import torch
import torch.nn as nn


class DilatedInception(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dilation: int = 1):
        super().__init__()
        branch = out_ch // 4
        self.k2 = nn.Conv2d(in_ch, branch, kernel_size=(1, 2), dilation=(1, dilation))
        self.k3 = nn.Conv2d(in_ch, branch, kernel_size=(1, 3), dilation=(1, dilation))
        self.k6 = nn.Conv2d(in_ch, branch, kernel_size=(1, 6), dilation=(1, dilation))
        self.k7 = nn.Conv2d(in_ch, branch, kernel_size=(1, 7), dilation=(1, dilation))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, N, T]
        y2, y3, y6, y7 = self.k2(x), self.k3(x), self.k6(x), self.k7(x)
        t_min = min(y2.size(-1), y3.size(-1), y6.size(-1), y7.size(-1))
        y = torch.cat([y2[..., -t_min:], y3[..., -t_min:], y6[..., -t_min:], y7[..., -t_min:]], dim=1)
        return y
