import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import hilbert2

from model import DCO
from utils import (
    apply_high_pass_filter,
    denormalize,
    fix_orit,
    normalize_data_i_h,
    normalize_data_input_raw,
)


class Model(nn.Module):
    def __init__(self, model_path: Path, device: str):
        super(Model, self).__init__()
        self.model_path = model_path
        dco = DCO()
        dco.load_state_dict(torch.load(self.model_path))

        def get_grid(shape):
            batchsize, size_x, size_y = shape[0], shape[1], shape[2]
            gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
            gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
            gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
            gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
            return torch.cat((gridx, gridy), dim=-1)

        self.model = dco
        self.rank = device if device != "auto" else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.rank)
        self.grid = get_grid((1, 256, 256)).to(self.rank)

    def __call__(self, dobs_300k, dobs_400k, dobs_500k):
        return self.forward(dobs_300k, dobs_400k, dobs_500k)

    def forward(self, dobs_300k, dobs_400k, dobs_500k):
        mask = np.load("./data/mask.npy")

        def h2f(x):
            return np.hstack(
                [
                    np.zeros((256, 29)),
                    np.abs(hilbert2(fix_orit(np.abs(x))[:, 29 : 256 - 28])),  # type: ignore
                    np.zeros((256, 28)),
                ]
            )

        def ampo(x):
            return fix_orit(np.abs(x))

        def aply_hp(x, y):
            return np.hstack(
                [
                    np.zeros((256, 29)),
                    apply_high_pass_filter(x[:, 29 : 256 - 28], y),
                    np.zeros((256, 28)),
                ]
            )

        dobs_300k = dobs_300k.squeeze() * mask
        dobs_400k = dobs_400k.squeeze() * mask
        dobs_500k = dobs_500k.squeeze() * mask
        k300_amp = h2f(dobs_300k)
        k300_amp_h = aply_hp(ampo(dobs_300k), 40)
        k300_ang = fix_orit(np.angle(dobs_300k))
        k400_amp = h2f(dobs_400k)
        k400_amp_h = aply_hp(ampo(dobs_400k), 40)
        k400_ang = fix_orit(np.angle(dobs_400k))
        k500_amp = h2f(dobs_500k)
        k500_amp_h = aply_hp(ampo(dobs_500k), 40)
        k500_ang = fix_orit(np.angle(dobs_500k))

        inputs_l = np.array([k300_amp, k300_ang, k400_amp, k400_ang, k500_amp, k500_ang]).astype(np.float32).reshape(6, 256, 256)
        inputs_l = normalize_data_input_raw(inputs_l).reshape(1, 6, 256, 256)
        inputs_l = torch.tensor(inputs_l).to(self.rank)
        inputs_h = np.array([k300_amp_h, k300_ang, k400_amp_h, k400_ang, k500_amp_h, k500_ang]).astype(np.float32).reshape(6, 256, 256)
        inputs_h = normalize_data_i_h(inputs_h).reshape(1, 6, 256, 256)
        inputs_h = torch.tensor(inputs_h).to(self.rank)
        pred_test = self.model(inputs_l, inputs_h, self.grid)
        pred_test = denormalize(pred_test)
        return pred_test


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-m", "--model", required=True, type=str, help="Path to the model")
    argparser.add_argument("-k3", required=True, type=str, help="Path to the 300k data")
    argparser.add_argument("-k4", required=True, type=str, help="Path to the 400k data")
    argparser.add_argument("-k5", required=True, type=str, help="Path to the 500k data")
    argparser.add_argument("-o", "--output", type=str, default=None, help="Path to the output")
    argparser.add_argument("--device", type=str, default="auto", help="Device to use")
    argparser.add_argument("-p", "--plot", action="store_true", default=False, help="Plot the output")

    args = argparser.parse_args()
    model = Model(args.model, args.device)
    dobs_300k = np.load(args.k3)
    dobs_400k = np.load(args.k4)
    dobs_500k = np.load(args.k5)
    pred = model(dobs_300k, dobs_400k, dobs_500k)
    pred = pred.cpu().detach().squeeze().numpy()
    if args.plot:
        import matplotlib.pyplot as plt

        plt.imshow(pred)
        plt.show()
    if args.output is not None:
        np.save(args.output, pred)
