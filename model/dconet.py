import torch
import torch.nn as nn
import torch.nn.functional as F

from .cno import CNO2d  # type: ignore
from .deeponet import FNN, DeepOnetNoBiasOrg  # type: ignore
from .fno import FNOBlock2d  # type: ignore


class DCO(nn.Module):
    def __init__(self, dropout_rate: float = 0.0):
        super(DCO, self).__init__()
        self.trunkl = FNN(2, [4, 8, 8, 10], 16)
        self.branchl = CNO2d(6, 16, 256, 4)
        self.deeponetl = DeepOnetNoBiasOrg(self.branchl, self.trunkl)
        self.trunkh = FNN(2, [4, 8, 8, 10], 16)
        self.branchh = CNO2d(6, 16, 256, 4)
        self.deeponeth = DeepOnetNoBiasOrg(self.branchh, self.trunkh)
        self.fc0 = nn.Linear(10, 12)
        self.fno0 = FNOBlock2d(12, 12, 32, 32)
        self.fno1 = FNOBlock2d(12, 16, 32, 32, dropout_rate=dropout_rate)
        self.fno2 = FNOBlock2d(16, 16, 32, 32, dropout_rate=dropout_rate)
        self.fno3 = FNOBlock2d(16, 16, 32, 32, dropout_rate=dropout_rate)
        self.fno4 = FNOBlock2d(16, 32, 32, 32, dropout_rate=dropout_rate)
        self.fno5 = FNOBlock2d(32, 32, 32, 32, dropout_rate=dropout_rate)
        self.fno6 = FNOBlock2d(32, 32, 64, 64, dropout_rate=dropout_rate)
        self.fno7 = FNOBlock2d(32, 32, 64, 64, dropout_rate=dropout_rate)
        self.fno8 = FNOBlock2d(32, 32, 128, 128, dropout_rate=dropout_rate)
        self.fno9 = FNOBlock2d(32, 32, 128, 128, dropout_rate=dropout_rate)

        self.output = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 32), nn.Linear(32, 1))

    def __call__(
        self,
        lf: torch.Tensor,
        hf: torch.Tensor,
        grid: torch.Tensor,
        hidden: bool = False,
    ):
        return self.forward(lf, hf, grid, hidden)

    def forward(
        self,
        lf: torch.Tensor,
        hf: torch.Tensor,
        grid: torch.Tensor,
        hidden: bool = False,
    ) -> torch.Tensor:
        # B, C, H, W = lf.shape
        x1 = self.deeponetl(lf, grid)
        x2 = self.deeponeth(hf, grid)

        x = torch.cat((x1, x2, grid), dim=-1)
        x = self.fc0(x)

        x = x.permute(0, 3, 1, 2)
        x = self.fno0(x, 256, 256)
        x = self.fno1(x, 256, 256)
        x = self.fno2(x, 256, 256)
        x = self.fno3(x, 256, 256)
        x = self.fno4(x, 256, 256)
        x = F.interpolate(x, size=(300, 300), mode="bilinear", align_corners=False)
        x = self.fno5(x, 300, 300)
        x = self.fno6(x, 300, 300)
        x = self.fno7(x, 300, 300)
        x = self.fno8(x, 300, 300)
        x = self.fno9(x, 300, 300)
        x = x.permute(0, 2, 3, 1)

        x = self.output(x)
        return x.squeeze(-1)
