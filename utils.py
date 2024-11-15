import math

import numpy as np
from scipy.fftpack import fft2, fftshift, ifft2, ifftshift

# max_data_input_raw: 1290.742919921875, min_data_input_raw: -3.1415624618530273
# max_data_i_h: 699.8890991210938, min_data_i_h: -703.486083984375
# max_speed_raw: 1603.64208984375, min_speed_raw: 1396.9390869140625
# max_speed_blur: 1570.9803466796875, min_speed_blur: 1421.4930419921875
# kmean: 285.3589172363281, kstd: 189.26708984375


def gaussian_low_pass_filter(shape, cutoff_freq):
    """
    生成二维高斯低通滤波器
    :param shape: 输入数据的形状 (h, w)
    :param cutoff_freq: 滤波器的截止频率（标准差）
    :return: 高斯低通滤波器
    """
    h, w = shape
    y, x = np.ogrid[:h, :w]
    center = (h // 2, w // 2)
    dist_square = (x - center[1]) ** 2 + (y - center[0]) ** 2
    gaussian_filter = np.exp(-dist_square / (2 * (cutoff_freq**2)))
    return gaussian_filter


def gaussian_high_pass_filter(shape, cutoff_freq):
    """
    生成二维高斯高通滤波器，通过 1 - 低通滤波器实现
    :param shape: 输入数据的形状 (h, w)
    :param cutoff_freq: 滤波器的截止频率（标准差）
    :return: 高斯高通滤波器
    """
    # 获取低通滤波器
    low_pass_filter = gaussian_low_pass_filter(shape, cutoff_freq)

    # 高通滤波器是 1 - 低通滤波器
    high_pass_filter = 1 - low_pass_filter
    return high_pass_filter


def apply_high_pass_filter(image, cutoff_freq):
    """
    对输入图像应用二维高斯高通滤波器
    :param image: 输入二维np.ndarray图像
    :param cutoff_freq: 截止频率
    :return: 滤波后的图像
    """
    # 获取图像大小
    h, w = image.shape

    # 生成高斯高通滤波器
    high_pass_filter = gaussian_high_pass_filter((h, w), cutoff_freq)

    # 对图像进行傅里叶变换
    image_fft = fft2(image)
    image_fft_shifted = fftshift(image_fft)  # type: ignore

    # 频域卷积（逐点相乘）
    filtered_fft_shifted = image_fft_shifted * high_pass_filter

    # 逆傅里叶变换回空间域
    filtered_fft = ifftshift(filtered_fft_shifted)
    filtered_image = ifft2(filtered_fft)

    # 取实部，因为结果中会有非常小的虚部
    filtered_image_real = np.real(filtered_image)  # type: ignore

    return filtered_image_real


def normalize_data_input_raw(data_input_raw):
    data_input_raw = np.array(data_input_raw)
    data_input_raw[0] = (data_input_raw[0] + 3.1415624618530273) / (1290.742919921875 + 3.1415624618530273)
    data_input_raw[1] = (data_input_raw[1] + math.pi) / (2 * math.pi)
    data_input_raw[2] = (data_input_raw[2] + 3.1415624618530273) / (1290.742919921875 + 3.1415624618530273)
    data_input_raw[3] = (data_input_raw[3] + math.pi) / (2 * math.pi)
    data_input_raw[4] = (data_input_raw[4] + 3.1415624618530273) / (1290.742919921875 + 3.1415624618530273)
    data_input_raw[5] = (data_input_raw[5] + math.pi) / (2 * math.pi)
    return data_input_raw


def normalize_data_i_h(data_i_h):
    data_i_h = np.array(data_i_h)
    # amp_layers = [0, 2, 4]
    # std = np.std(data_i_h[:,amp_layers])
    # mean = np.mean(data_i_h[:,amp_layers])
    # print(f"mean: {mean}, std: {std}")
    std = 27.314180374145508
    mean = 1.1484155987284694e-10
    data_i_h[0] = (data_i_h[0] - mean) / std
    data_i_h[1] = (data_i_h[1] + math.pi) / (2 * math.pi)
    data_i_h[2] = (data_i_h[2] - mean) / std
    data_i_h[3] = (data_i_h[3] + math.pi) / (2 * math.pi)
    data_i_h[4] = (data_i_h[4] - mean) / std
    data_i_h[5] = (data_i_h[5] + math.pi) / (2 * math.pi)
    return data_i_h


def denormalize(tensor):
    return tensor * (29.908472061157227 + 1e-16) + 1485.4248046875


def apply_low_pass_filter(image, cutoff_freq):
    """
    对输入图像应用二维高斯低通滤波器
    :param image: 输入二维np.ndarray图像
    :param cutoff_freq: 截止频率
    :return: 滤波后的图像
    """
    h, w = image.shape

    gaussian_filter = gaussian_low_pass_filter((h, w), cutoff_freq)

    image_fft = fft2(image)
    image_fft_shifted = fftshift(image_fft)  # type: ignore

    filtered_fft_shifted = image_fft_shifted * gaussian_filter

    filtered_fft = ifftshift(filtered_fft_shifted)
    filtered_image = ifft2(filtered_fft)

    filtered_image_real = np.real(filtered_image)  # type: ignore

    return filtered_image_real


def fix_orit(x: np.ndarray):
    out = np.zeros(x.shape)
    for i in range(x.shape[0]):
        out[i] = np.roll(x[i], -i)
    return out
