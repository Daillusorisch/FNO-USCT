import argparse
import os
from dataclasses import dataclass
from typing import Literal
from warnings import filterwarnings

import matplotlib.pyplot as plt
import numpy as np
import tomllib
import torch
import torch.optim.adam as optim
from scipy.signal import hilbert2
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter  # type: ignore
from tqdm.auto import tqdm

from loss import LapLoss, PSNRLoss, SSIMLoss
from model import DCO
from utils import apply_high_pass_filter, denormalize, fix_orit, normalize_data_i_h, normalize_data_input_raw

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("medium")

# disable torch load warnings
filterwarnings("ignore", category=FutureWarning)

CKPT_DIR = "./ckpt"


@dataclass
class TrainConfig:
    batch_size: int = 1
    learning_rate: float = 1e-3
    num_epochs: int = 1000
    data_dir: str = "./data"
    data_size: int = 7200
    ckpt_dir: str = CKPT_DIR
    save_interval: int = 10
    eval_interval: int = 5


def prepare_data(k300: np.ndarray, k400: np.ndarray, k500: np.ndarray):
    """
    deal with masked data
    """

    def h2f(x):
        return np.hstack([np.zeros((256, 29)), np.abs(hilbert2(fix_orit(np.abs(x))[:, 29 : 256 - 28])), np.zeros((256, 28))])  # type: ignore

    def aply_hp(x, y):
        return np.hstack([np.zeros((256, 29)), apply_high_pass_filter(x[:, 29 : 256 - 28], y), np.zeros((256, 28))])

    k300_amp = h2f(k300)
    k300_amp_h = aply_hp(k300, 40)
    k300_ang = fix_orit(np.angle(k300))
    k400_amp = h2f(k400)
    k400_amp_h = aply_hp(k400, 40)
    k400_ang = fix_orit(np.angle(k400))
    k500_amp = h2f(k500)
    k500_amp_h = aply_hp(k500, 40)
    k500_ang = fix_orit(np.angle(k500))
    inputl = np.stack([k300_amp, k300_ang, k400_amp, k400_ang, k500_amp, k500_ang], axis=0).astype(np.float32)
    inputl = normalize_data_input_raw(inputl)
    inputh = np.stack([k300_amp_h, k300_ang, k400_amp_h, k400_ang, k500_amp_h, k500_ang], axis=0).astype(np.float32)
    inputh = normalize_data_i_h(inputh)
    return [inputl, inputh]


class DobsDataset(Dataset):
    def __init__(self, k300: np.ndarray, k400: np.ndarray, k500: np.ndarray, speed: np.ndarray):
        self.k300 = k300
        self.k400 = k400
        self.k500 = k500
        self.speed = speed
        self.length = len(k300)

    def add_awgn(self, signal, snr_dB):
        # 计算信号的功率
        signal_power = np.mean(signal**2)
        # 计算噪声的功率
        snr_linear = 10 ** (snr_dB / 10.0)
        noise_power = signal_power / snr_linear
        # 生成高斯白噪声
        noise = np.sqrt(noise_power) * np.random.randn(*signal.shape)
        # 将噪声添加到信号中
        noisy_signal = signal + noise
        return noisy_signal

    def __getitem__(self, idx):
        k300 = self.k300[idx]
        k400 = self.k400[idx]
        k500 = self.k500[idx]
        spd = self.speed[idx]
        rand = np.random.randint(5, 30)
        if np.random.rand() > 0.3:
            k300 = self.add_awgn(k300, rand)
            k400 = self.add_awgn(k400, rand)
            k500 = self.add_awgn(k500, rand)
        low, high = prepare_data(k300, k400, k500)

        return low, high, spd

    def __len__(self):
        return self.length


def plot_pred_real(real, pred, input):
    if isinstance(input, torch.Tensor):
        input = input.cpu().detach().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().detach().numpy()
    if isinstance(real, torch.Tensor):
        real = real.cpu().detach().numpy()
    # print(real.shape, pred.shape, input.shape)
    fig = plt.figure(figsize=(16, 5))
    ax0 = fig.add_subplot(1, 3, 1)
    ax0.imshow(input[0])
    ax0.set_title("input")
    ax0.axis("off")
    ax1 = fig.add_subplot(1, 3, 2)
    ax1.imshow(pred)
    ax1.set_title("pred")
    ax1.axis("off")
    fig.colorbar(ax1.imshow(pred), ax=ax1)
    ax2 = fig.add_subplot(1, 3, 3)
    ax2.imshow(real)
    ax2.set_title("true")
    ax2.axis("off")
    fig.colorbar(ax2.imshow(real), ax=ax2)
    fig.tight_layout()
    return fig


def get_grid(shape):
    batchsize, size_x, size_y = shape[0], shape[1], shape[2]
    gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
    gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
    gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
    gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
    return torch.cat((gridx, gridy), dim=-1)


class Trainer:
    model: DCO
    optimizer: optim.Adam
    train_loader: DataLoader[tuple[np.ndarray, np.ndarray, np.ndarray]]
    test_loader: DataLoader[tuple[np.ndarray, np.ndarray, np.ndarray]]
    epoch: int  # epoch
    steps: int  # steps
    config: TrainConfig
    rank: int | str
    writer: SummaryWriter

    def __init__(self, model: DCO, optimizer: optim.Adam, config: TrainConfig, rank):
        model = model.to(rank)
        self.grid = get_grid((config.batch_size, 256, 256)).to(rank)
        self.test_grid = get_grid((1, 256, 256)).to(rank)
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self.writer = SummaryWriter()
        train_loader, test_loader = self.load_data()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epoch = 0
        self.steps = 0
        self.test_steps = 0

        def load_evaldata(num, use: Literal["train", "pred"]):
            loader = self.train_loader if use == "train" else self.test_loader
            data1 = loader.dataset.k300[num]  # type: ignore
            data2 = loader.dataset.k400[num]  # type: ignore
            data3 = loader.dataset.k500[num]  # type: ignore
            data4 = loader.dataset.speed[num]  # type: ignore
            low, high = prepare_data(data1, data2, data3)

            low = torch.from_numpy(low).to(rank)
            high = torch.from_numpy(high).to(rank)
            spd = torch.from_numpy(data4).to(rank)

            return (low, high, spd)

        self.eval_datas = {
            "train1": load_evaldata(3, "train"),
            "train2": load_evaldata(930, "train"),
            "pred1": load_evaldata(1, "pred"),
            "pred2": load_evaldata(15, "pred"),
            "pred3": load_evaldata(70, "pred"),
            "pred4": load_evaldata(110, "pred"),
        }

        self.rank = rank
        self.psnr_loss = PSNRLoss(data_range=(1396.9390869140625, 1603.64208984375)).to(rank)
        self.l1_loss = torch.nn.L1Loss().to(rank)
        self.ssim_loss = SSIMLoss(kernel_size=11, data_range=(1396.9390869140625, 1603.64208984375)).to(rank)
        self.lap_loss = LapLoss().to(rank)

    def load_data(self):
        data_size = self.config.data_size
        data_dir = self.config.data_dir
        dobs_train_set: DobsDataset
        dobs_test_set: DobsDataset

        mask = np.load(os.path.join(data_dir, "mask.npy"))
        k300 = np.array([np.load(os.path.join(data_dir, f"dobs_300k_train/train_{int(i)}.npy")) * mask for i in range(1, data_size + 1)])
        k400 = np.array([np.load(os.path.join(data_dir, f"dobs_300k_train/train_{int(i)}.npy")) * mask for i in range(1, data_size + 1)])
        k500 = np.array([np.load(os.path.join(data_dir, f"dobs_300k_train/train_{int(i)}.npy")) * mask for i in range(1, data_size + 1)])
        speed_raw = np.array([np.load(os.path.join(data_dir, f"speed_train_fix/train_{int(i)}.npy")) for i in range(1, data_size + 1)])

        k300_train = k300[: data_size * 4 // 5, :, :]
        k300_test = k300[data_size * 4 // 5 :, :, :]
        k400_train = k400[: data_size * 4 // 5, :, :]
        k400_test = k400[data_size * 4 // 5 :, :, :]
        k500_train = k500[: data_size * 4 // 5, :, :]
        k500_test = k500[data_size * 4 // 5 :, :, :]
        speed_train = speed_raw[: data_size * 4 // 5, :, :]
        speed_test = speed_raw[data_size * 4 // 5 :, :, :]

        dobs_train_set = DobsDataset(k300_train, k400_train, k500_train, speed_train)
        dobs_test_set = DobsDataset(k300_test, k400_test, k500_test, speed_test)

        train_loader = DataLoader(dobs_train_set, batch_size=self.config.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(dobs_test_set, batch_size=self.config.batch_size, shuffle=False, drop_last=True)

        return train_loader, test_loader

    def loss_fn(self, pred_speed, speed, stage: Literal["train", "test"] = "train"):
        # TODO: denormalize?
        pred_speed = pred_speed.unsqueeze(1)
        speed = speed.unsqueeze(1)
        loss_l = self.lap_loss(pred_speed, speed)
        loss_m = self.l1_loss(pred_speed, speed)
        loss_p = self.psnr_loss(pred_speed, speed)
        loss_s = 1 - self.ssim_loss(pred_speed, speed)
        # loss_g = self.gdl_loss(pred_speed, speed)
        loss_s = loss_s * 100
        loss_l = loss_l * 0.5

        step: int
        match stage:
            case "train":
                step = self.steps
            case "test":
                step = self.test_steps
        self.writer.add_scalar(f"LapLoss/{stage}", loss_l, step)
        self.writer.add_scalar(f"PSNRLoss/{stage}", loss_p, step)
        self.writer.add_scalar(f"SSIMLoss/{stage}", loss_s, step)
        self.writer.add_scalar(f"L1Loss/{stage}", loss_m, step)
        # writer.add_scalar(f"GDLLoss/{stage}", loss_g, step)

        return loss_l + loss_s - loss_p * 0.5 + loss_m

    # on rank
    def _step(self, low, high, speed):
        self.optimizer.zero_grad()
        pred_speed = denormalize(self.model(low, high, self.grid))
        loss = self.loss_fn(pred_speed, speed)
        loss.backward()
        self.optimizer.step()

    def a_ephoc(self):
        self.model.train()
        for low, high, speed in self.train_loader:
            low = low.to(self.rank)
            high = high.to(self.rank)
            speed = speed.to(self.rank)
            self._step(low, high, speed)
            del low, high, speed
            self.steps += 1

    @torch.no_grad
    def test(self):
        self.model.eval()
        for low, high, speed in self.test_loader:
            low = low.to(self.rank)
            high = high.to(self.rank)
            speed = speed.to(self.rank)
            pred_speed = self.model(low, high, self.grid)
            pred_speed = denormalize(pred_speed)
            _ = self.loss_fn(pred_speed, speed, stage="test")
            self.test_steps += 1

    @torch.no_grad
    def eval(self):
        self.model.eval()
        for key, data in self.eval_datas.items():
            low, high, speed = data
            pred_speed = self.model(low.unsqueeze(0), high.unsqueeze(0), self.test_grid)
            pred_speed = denormalize(pred_speed)
            self.writer.add_figure(f"eval/{key}", plot_pred_real(speed.squeeze(), pred_speed.squeeze(), low.squeeze()), self.epoch)
            del low, high, speed, pred_speed

    def histogram(self):
        pass

    def save_checkpoint(self):
        ckpt_dir = self.config.ckpt_dir
        if not os.path.exists(ckpt_dir):
            os.mkdir(ckpt_dir)
        path = os.path.join(ckpt_dir, f"checkpoint_{self.epoch}-{self.steps}.pth")
        torch.save(self.model.state_dict(), path)

    def train(self):
        temp = self.epoch
        self.epoch = -1
        self.eval()
        self.epoch = temp
        for epoch in tqdm(range(self.config.num_epochs)):
            self.a_ephoc()
            self.test()
            if epoch % self.config.save_interval == 0:
                self.save_checkpoint()
            if epoch % self.config.eval_interval == 0:
                self.eval()
                # self.histogram()
            self.epoch += 1

        # save the model after training
        self.save_checkpoint()
        self.writer.close()


def load_ckpt(ckpt_dir: str, model: DCO, static: bool = True) -> tuple[DCO, int, int]:
    start_epoch = 0
    start_step = 0
    if os.path.exists(ckpt_dir):
        ckpt = os.listdir(ckpt_dir)
        if ckpt.__len__() == 0:
            print(f"no checkpoint found in {ckpt_dir}")
            return model, start_epoch, start_step
        epochs = [int(epc.split("_")[1].split("-")[0]) for epc in ckpt]
        steps = [int(stp.split("_")[1].split("-")[1].split(".")[0]) for stp in ckpt]
        if epochs:
            epochs.sort()
            steps.sort()
            start_epoch = epochs[-1]
            start_step = steps[-1]
            path = os.path.join(ckpt_dir, f"checkpoint_{start_epoch}-{start_step}.pth")
            print(f"loading checkpoint_{start_epoch}-{start_step}.pth from {ckpt_dir}")
            if not static:
                net_dict = model.state_dict()
                predict_model = torch.load(path)
                print("using unstatic loading")

                state_dict = {k: v for k, v in predict_model.items() if k in net_dict.keys()}  # find public keys
                net_dict.update(state_dict)
                model.load_state_dict(net_dict)
            else:
                model.load_state_dict(torch.load(path))
    return model, start_epoch, start_step


def main(config_path: str | None = None):
    model = DCO(dropout_rate=0.04)
    config = TrainConfig()
    if config_path is None:
        with open("config.toml", "rb") as f:
            config_toml = tomllib.load(f)
        config = TrainConfig(**config_toml["train"])
    else:
        with open(config_path, "rb") as f:
            config_toml = tomllib.load(f)
        config = TrainConfig(**config_toml["train"])
    model, start_epoch, start_step = load_ckpt(config.ckpt_dir, model)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    trainer = Trainer(model, optimizer, config, "cuda")  # change to "cpu" if you don't have a gpu
    trainer.steps = start_step
    trainer.epoch = start_epoch
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default=None, help="path to the config file")
    args = parser.parse_args()
    main(args.config)
