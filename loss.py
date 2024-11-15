import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class SSIMLoss(torch.nn.Module):
    def __init__(
        self, kernel_size: int = 11, data_range: tuple[float, float] = (1396.9390869140625, 1603.64208984375), average: bool = True
    ):
        super(SSIMLoss, self).__init__()
        self.data_range = data_range[1] - data_range[0]
        self.average = average
        self.kernel_size = kernel_size
        self.kernel = nn.Parameter(build_gauss_kernel(kernel_size), requires_grad=False)

    def forward(self, preds: torch.Tensor, target: torch.Tensor):
        img1 = preds.to(torch.float32)
        img2 = target.to(torch.float32)

        # Squares of input matrices
        K1 = 0.01
        K2 = 0.03
        C1 = (K1 * self.data_range) ** 2
        C2 = (K2 * self.data_range) ** 2

        channel = img1.size(1)
        window = self.kernel
        window_size = self.kernel_size

        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if self.average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)


class PSNRLoss(torch.nn.Module):
    def __init__(self, data_range: tuple[float, float]):
        super(PSNRLoss, self).__init__()
        self.data_range = data_range

    def forward(self, preds, target):
        mse = F.mse_loss(preds, target, reduction="mean")
        psnr = 10 * torch.log10(self.data_range[1] / mse)
        return psnr


class LapLoss(nn.Module):
    def __init__(self, max_levels=5):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self._gauss_kernel = None

    def forward(self, input, target):
        if self._gauss_kernel is None or self._gauss_kernel.shape[1] != input.shape[1]:
            self._gauss_kernel = gauss_kernel5(input.shape[1], cuda=input.is_cuda)

        pyr_input = laplacian_pyramid_expand(input, self._gauss_kernel, self.max_levels)
        pyr_target = laplacian_pyramid_expand(target, self._gauss_kernel, self.max_levels)
        weights = [1, 2, 4, 8, 16]

        # return sum(F.l1_loss(a, b) for a, b in zip(pyr_input, pyr_target))
        return sum(weights[i] * F.l1_loss(a, b) for i, (a, b) in enumerate(zip(pyr_input, pyr_target))).mean()  # type: ignore


def pyr_downsample(x):
    return x[:, :, ::2, ::2]


def pyr_upsample(x, kernel, op0, op1):
    n_channels, _, kw, kh = kernel.shape
    return F.conv_transpose2d(x, kernel, groups=n_channels, stride=2, padding=2, output_padding=(op0, op1))


def gauss_kernel5(channels=3, cuda=True):
    kernel = torch.FloatTensor(
        [
            [1.0, 4.0, 6.0, 4.0, 1],
            [4.0, 16.0, 24.0, 16.0, 4.0],
            [6.0, 24.0, 36.0, 24.0, 6.0],
            [4.0, 16.0, 24.0, 16.0, 4.0],
            [1.0, 4.0, 6.0, 4.0, 1.0],
        ]
    )
    kernel /= 256.0
    kernel = kernel.repeat(channels, 1, 1, 1)
    # print(kernel)
    if cuda:
        kernel = kernel.cuda()
    return Variable(kernel, requires_grad=False)


def build_gauss_kernel(size=5, sigma=1.0, n_channels=1, cuda=False):
    if size % 2 != 1:
        raise ValueError("kernel size must be uneven")
    grid = np.float32(np.mgrid[0:size, 0:size].T)

    def gaussian(x):
        return np.exp((x - size // 2) ** 2 / (-2 * sigma**2)) ** 2

    kernel = np.sum(gaussian(grid), axis=2)
    kernel /= np.sum(kernel)
    # repeat same kernel across depth dimension
    kernel = np.tile(kernel, (n_channels, 1, 1))
    # conv weight should be (out_channels, groups/in_channels, h, w),
    # and since we have depth-separable convolution we want the groups dimension to be 1
    kernel = torch.FloatTensor(kernel[:, None, :, :])
    # kernel = gauss_kernel5(n_channels)
    if cuda:
        kernel = kernel.cuda()
    return Variable(kernel, requires_grad=False)


def conv_gauss(img, kernel):
    """convolve img with a gaussian kernel that has been built with build_gauss_kernel"""
    n_channels, _, kw, kh = kernel.shape
    img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode="replicate")
    return F.conv2d(img, kernel, groups=n_channels)


def laplacian_pyramid(img, kernel, max_levels=5):
    current = img
    pyr = []

    for level in range(max_levels - 1):
        filtered = conv_gauss(current, kernel)
        diff = current - filtered
        pyr.append(diff)
        current = F.avg_pool2d(filtered, 2)

    pyr.append(current)  # high -> low
    return pyr


def laplacian_pyramid_expand(img, kernel, max_levels=5):
    current = img
    pyr = []
    for level in range(max_levels):
        # print("level: ", level)
        filtered = conv_gauss(current, kernel)
        down = pyr_downsample(filtered)
        up = pyr_upsample(down, 4 * kernel, 1 - filtered.size(2) % 2, 1 - filtered.size(3) % 2)

        diff = current - up
        pyr.append(diff)

        current = down
    return pyr
