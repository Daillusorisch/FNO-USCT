import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(Linear2d, self).__init__()
        self.w = nn.Parameter(torch.randn(in_channels, out_channels))
        self.b = nn.Parameter(torch.randn(out_channels))

    def forward(self, x: torch.Tensor):
        # print(x.shape, self.w.shape)
        return torch.einsum("bixy,io->boxy", x, self.w) + self.b[:, None, None]


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mode_x: int, mode_y: int):
        super(SpectralConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mode_x = mode_x
        self.mode_y = mode_y

        self.scale = (1 / (2 * in_channels)) ** (1.0 / 2.0)
        self.weights1 = nn.Parameter(
            self.scale
            * (
                torch.randn(
                    in_channels,
                    out_channels,
                    self.mode_x,
                    self.mode_y,
                    dtype=torch.complex64,
                )
            )
        )
        self.weights2 = nn.Parameter(
            self.scale
            * (
                torch.randn(
                    in_channels,
                    out_channels,
                    self.mode_x,
                    self.mode_y,
                    dtype=torch.complex64,
                )
            )
        )

    # Complex multiplication
    def compl_mul2d(self, x_h, weights):
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        return torch.einsum("bixy,ioxy->boxy", x_h, weights)

    def forward(self, x: torch.Tensor, dim_x: int | None = None, dim_y: int | None = None) -> torch.Tensor:
        if dim_x is None:
            dim_x = x.shape[-2]
        if dim_y is None:
            dim_y = x.shape[-1]
        # x.shape (b, c, x, y)
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_h = torch.fft.rfft2(x, norm="forward")

        out = torch.zeros(
            batchsize,
            self.out_channels,
            dim_x,
            dim_y // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out[:, :, : self.mode_x, : self.mode_y] = self.compl_mul2d(x_h[:, :, : self.mode_x, : self.mode_y], self.weights1)
        out[:, :, -self.mode_x :, : self.mode_y] = self.compl_mul2d(x_h[:, :, -self.mode_x :, : self.mode_y], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out, s=(dim_x, dim_y), norm="forward")
        return x


class WOp2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(WOp2d, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor, dim_x: int, dim_y: int):
        x_out = self.conv(x)
        x_out = torch.nn.functional.interpolate(
            x_out,
            size=(dim_x, dim_y),
            mode="bicubic",
            align_corners=True,
            antialias=True,
        )
        return x_out


class FNOBlock2d(nn.Module):
    """
    input shape (b, ci, x, y)
    output shape (b, co, dimx, dimy)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode_x: int,
        mode_y: int,
        dropout_rate: float = 0.0,
    ):
        super(FNOBlock2d, self).__init__()
        self.conv = SpectralConv2d(in_channels, out_channels, mode_x, mode_y)
        self.w = WOp2d(in_channels, out_channels)
        self.normalize = torch.nn.InstanceNorm2d(out_channels, affine=True)
        self.dropout = nn.Dropout2d(dropout_rate)

    def __call__(self, x: torch.Tensor, dim_x: int, dim_y: int):
        return self.forward(x, dim_x, dim_y)

    def forward(self, x: torch.Tensor, dim_x: int, dim_y: int):
        x1 = self.conv(x, dim_x, dim_y)
        x2 = self.w(x, dim_x, dim_y)
        out = x1 + x2
        out = self.normalize(out)
        out = F.gelu(out)
        out = self.dropout(out)
        return out


class UNO(nn.Module):
    def __init__(self, in_width: int, width: int, pad=8):
        super(UNO, self).__init__()

        # input function co-domain dimention after concatenating (x,y)
        self.in_width = in_width
        self.width = width  # lifting dimension

        self.padding = pad  # passing amount

        self.linear0 = Linear2d(self.in_width, self.width // 2)
        self.linear1 = Linear2d(self.width // 2, self.width)

        self.fno0 = FNOBlock2d(self.width, 2 * self.width, 32, 32)
        self.fno1 = FNOBlock2d(2 * self.width, 4 * self.width, 16, 16)
        self.fno2 = FNOBlock2d(4 * self.width, 8 * self.width, 8, 8)
        self.fno3 = FNOBlock2d(8 * self.width, 16 * self.width, 4, 4)

        self.fno4 = FNOBlock2d(16 * self.width, 16 * self.width, 4, 4)
        # self.fno5  = FNOBlock2d(16*self.width, 16*self.width,  4,  4)
        # self.fno6  = FNOBlock2d(16*self.width, 16*self.width,  4,  4)
        self.fno7 = FNOBlock2d(16 * self.width, 16 * self.width, 4, 4)
        self.fno8 = FNOBlock2d(16 * self.width, 16 * self.width, 4, 4)

        self.fno9 = FNOBlock2d(16 * self.width, 8 * self.width, 8, 8)
        self.fno10 = FNOBlock2d(16 * self.width, 4 * self.width, 16, 16)
        self.fno11 = FNOBlock2d(8 * self.width, 2 * self.width, 32, 32)
        self.fno12 = FNOBlock2d(4 * self.width, self.width, 64, 64)
        self.fno13 = FNOBlock2d(1 * self.width, self.width // 2, 64, 64)

        self.linear2 = Linear2d(self.width // 2, 2 * self.width)
        self.linear3 = Linear2d(2 * self.width, 1)

    def forward(self, x: torch.Tensor, scale_factor: float = 1.0):
        # input (b, c, x, y)
        x = self.linear0(x)
        x = F.gelu(x)

        x = self.linear1(x)
        x = F.gelu(x)

        # x_fc0 = F.pad(x_fc0, [0, self.padding, 0, self.padding])

        # (b, x, y, c)
        D1, D2 = x.shape[-2], x.shape[-1]
        # print(D1, D2)

        x_c0 = self.fno0(x, D1 // 2, D2 // 2)
        x_c1 = self.fno1(x_c0, D1 // 4, D2 // 4)
        x_c2 = self.fno2(x_c1, D1 // 8, D2 // 8)
        x_c3 = self.fno3(x_c2, D1 // 16, D2 // 16)

        x = self.fno4(x_c3, D1 // 16, D2 // 16)
        # x = self.fno5(x, D1//16, D2//16)
        # x = self.fno6(x, D1//16, D2//16)
        x = self.fno7(x, D1 // 16, D2 // 16)
        x = self.fno8(x, D1 // 16, D2 // 16)

        x_c9 = self.fno9(x, D1 // 8, D2 // 8)
        x_c9 = torch.cat([x_c9, x_c2], dim=1)

        x_c10 = self.fno10(x_c9, D1 // 4, D2 // 4)
        x_c10 = torch.cat([x_c10, x_c1], dim=1)

        x_c11 = self.fno11(x_c10, D1 // 2, D2 // 2)
        x_c11 = torch.cat([x_c11, x_c0], dim=1)

        x_c12 = self.fno12(x_c11, D1, D2)

        x_c13 = self.fno13(x_c12, int(D1 * scale_factor), int(D2 * scale_factor))

        x = self.linear2(x_c13)
        x = F.gelu(x)
        x = self.linear3(x)

        return x


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(90, 6, 256, 256).to(device)
    model = UNO(6, 16).to(device)
    out = model(x, 1.875)
    print(out.shape)
