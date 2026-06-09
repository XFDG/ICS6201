import torch
import torch.nn as nn
import torch.nn.functional as F


class ECA(nn.Module):
    def __init__(self, channels: int, k_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)
        return x * y


class BiFPNFuse(nn.Module):
    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.c1 = int(c1)
        self.c2 = int(c2)
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.eps = 1e-4
        c_out = self.c1 + self.c2
        self.out = nn.Sequential(
            nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = F.relu(self.w)
        w = w / (w.sum() + self.eps)
        x1 = x[:, : self.c1]
        x2 = x[:, self.c1 : self.c1 + self.c2]
        y = torch.cat((w[0] * x1, w[1] * x2), dim=1)
        return self.out(y)
