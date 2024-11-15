from typing import List, Sequence

import torch
import torch.nn as nn


class Premute(nn.Module):
    def __init__(self, out_shape: List[int]):
        super(Premute, self).__init__()
        self.out_shape = out_shape

    def forward(self, x: torch.Tensor):
        return x.permute(self.out_shape)


class FNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], output_dim: int):
        super(FNN, self).__init__()
        self.input_dim = input_dim
        self.out_dim = output_dim
        self.hidden_dims = hidden_dims

        self.layers = []
        in_dim = input_dim

        count = 0
        for h_dim in hidden_dims:
            self.layers.append(nn.Linear(in_dim, h_dim))
            if count == 4:
                self.layers.append(Premute([0, 3, 1, 2]))
                self.layers.append(nn.GroupNorm(2, h_dim))
                self.layers.append(Premute([0, 2, 3, 1]))
                count = 0
            self.layers.append(nn.GELU())
            in_dim = h_dim
            count += 1
        self.layers.append(nn.Linear(in_dim, output_dim))
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)


class DeepOnetNoBiasOrg(nn.Module):
    def __init__(self, branch, trunk):
        super(DeepOnetNoBiasOrg, self).__init__()
        self.branch = branch
        self.trunk = trunk
        self.b0 = torch.nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.p = self.trunk.out_dim

    def forward(self, u_: torch.Tensor, x_: torch.Tensor):
        # print(x_.shape)
        weights = self.branch(u_)
        # (b, c, x, y) -> (b, x, y, c)
        weights = weights.permute(0, 2, 3, 1)
        basis = self.trunk(x_)

        if not isinstance(self.trunk, FNN):
            basis = basis.permute(0, 2, 3, 1)
            # print(basis.shape)

        ch = basis.shape[-1]
        output = torch.stack(
            [
                torch.einsum("bxyc,bxyc->bxy", weights[..., 0 : ch // 4], basis[..., 0 : ch // 4]),
                torch.einsum(
                    "bxyc,bxyc->bxy",
                    weights[..., ch // 4 : ch // 2],
                    basis[..., ch // 4 : ch // 2],
                ),
                torch.einsum(
                    "bxyc,bxyc->bxy",
                    weights[..., ch // 2 : ch // 4 * 3],
                    basis[..., ch // 2 : ch // 4 * 3],
                ),
                torch.einsum(
                    "bxyc,bxyc->bxy",
                    weights[..., ch // 4 * 3 : ch],
                    basis[..., ch // 4 * 3 : ch],
                ),
            ],
            dim=-1,
        )

        return (output + self.b0) / self.p**0.5
