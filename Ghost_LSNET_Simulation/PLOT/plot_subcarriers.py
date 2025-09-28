import gc
from scipy.signal import wiener

from LSNET.LSNet import load_channel_data
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # 新增进度条库

import torch
import torch.nn as nn

from LSNET.LSNet import load_channel_data, CustomDataset, DataLoader, LSNet_1D
from GHOST_LSNET.Ghost_LSNet import LSGhostNet_1D, get_evm_func_for_mid
from DNCNN.DNCNN import DNCNN
from utils import WIENER_FILTER_LOAD
from plot_EVM import  Baseline_MLP
# 计算均方误差（MSE）
def calculate_mse(original, filtered):
    return np.mean((original - filtered) ** 2)

def WIENER_FILTER_LOAD(dbvalue, channel_name):
    x, y = load_channel_data("D:/AI_Filter/", dbvalue, 10, 1, [channel_name], '4x80')

    num_samples = 8000
    num_signals = 1001
    tperfectChannel = y.reshape(num_samples, num_signals)
    tnoisyChannel = x.reshape(num_samples, num_signals)

    mmse_opt = calculate_mse(tperfectChannel, tnoisyChannel)
    print("mmse_opt", mmse_opt)
    optsize = 0

    for size in range(3,20):
        # 应用维纳滤波，对每一条数据单独进行滤波
        print("size test:", size)
        filtered_signal = np.apply_along_axis(lambda x: wiener(x, mysize=size), 1, tnoisyChannel)
        mmse = calculate_mse(tperfectChannel, filtered_signal)
        if mmse < mmse_opt:
            mmse_opt = mmse
            optsize = size
            print("mmse", mmse, "mmse_opt", mmse_opt, "optsize", optsize)

    y = np.apply_along_axis(lambda x: wiener(x, mysize=optsize), 1, tnoisyChannel)
    x = tnoisyChannel

    print(y.shape)
    # 初始化 SNR 数组
    snr_per_subcarrier = np.zeros((1001))

    # 计算每个子载波的 SNR
    for subcarrier in range(1001):
        signal_power_subcarrier = np.mean(np.abs(x[:, subcarrier]) ** 2)
        noise_power_subcarrier = np.mean(
            np.abs(x[:, subcarrier] - y[:, subcarrier]) ** 2)

        snr_per_subcarrier[subcarrier] = 10 * np.log10(signal_power_subcarrier / noise_power_subcarrier)

    from scipy.signal import medfilt
    filtered_data = medfilt(snr_per_subcarrier[10:-10], kernel_size=55)  # 尝试窗口大小 3，可根据效果调整
    del x, y
    collected = gc.collect()
    print(f"回收对象数：{collected}")
    return np.concatenate((snr_per_subcarrier[:10], filtered_data, snr_per_subcarrier[-10:]))


def GLSNET_EVALUATE(db_value, model_name, channel_name, DLNET):
    x, y = load_channel_data("D:/AI_Filter/", db_value, 10, 1, [channel_name], '4x80')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检查是否有可用的GPU
    next_x_test = torch.from_numpy(x).float().to(device)
    next_y_test = torch.from_numpy(y).float().to(device)
    next_test_dataset = CustomDataset(next_x_test, next_y_test)
    test_loader = DataLoader(next_test_dataset, batch_size=64, shuffle=False, num_workers=0)
    all_p = []

    checkpoint_save_path = f"../{model_name}/checkpoint/{channel_name}/{model_name}_margin10_num10_epoch200_model{DLNET.name}_setting4x80_dB{db_value}_channel{channel_name}.weights.pth"

    DLNET.load_state_dict(torch.load(checkpoint_save_path, weights_only=True))

    mode = 'MLP'

    # 测试阶段
    DLNET.eval()



    checkpoint_save_path1 = f"../{model_name}/checkpoint/{channel_name}/num10epoch2000_Baseline_MLP_4x80_{db_value}dB_{channel_name}_margin10.weights.pth"
    checkpoint_save_path2 = f"../{model_name}/checkpoint/{channel_name}/num10epoch2000_Baseline_MLP_4x80_{db_value}dB_{channel_name}_margin-10.weights.pth"
    mlp1 = Baseline_MLP().to(device)
    mlp2 = Baseline_MLP().to(device)
    mlp1.load_state_dict(torch.load(checkpoint_save_path1, weights_only=True))
    mlp2.load_state_dict(torch.load(checkpoint_save_path2, weights_only=True))

    mlp1.eval()
    mlp2.eval()

    with torch.no_grad():
        for inputs, labels in test_loader:

            output1 = mlp1(inputs[:, :, :20])
            output2 = DLNET(inputs)
            output3 = mlp2(inputs[:, :, -20:])
            batch_p = torch.cat((output1, output2[:, :, 10:-10], output3), dim=2)

            all_p.append(batch_p.cpu().detach())


    p = torch.cat(all_p, dim=0)
    data_len = 8000

    x_test = x.reshape(data_len, 1001, 1)
    y_test = y.reshape(data_len, 1001, 1)
    p_test = p.numpy().reshape(data_len, 1001, 1)
    loss = x_test - y_test

    scidx4x = np.hstack((range(-500, -2, 1), range(3, 501, 1)))
    indices = scidx4x + 500
    get_evm_func_for_mid(indices[:], p_test, x_test, y_test)

    p_test = p.numpy().reshape(8000, 1001)
    x = x.reshape(8000, 1001)


    import gc
    collected = gc.collect()
    print(f"回收对象数：{collected}")

    x = x - p_test
    y = y.reshape(8000, 1001)
    # 初始化 SNR 数组
    snr_per_subcarrier = np.zeros((1001))

    # 计算每个子载波的 SNR
    for subcarrier in range(1001):
        signal_power_subcarrier = np.mean(np.abs(x[:, subcarrier]) ** 2)
        noise_power_subcarrier = np.mean(
            np.abs(x[:, subcarrier]-y[:, subcarrier]) ** 2)

        snr_per_subcarrier[subcarrier] = 10 * np.log10(signal_power_subcarrier / noise_power_subcarrier)

    from scipy.signal import medfilt
    filtered_data = medfilt(snr_per_subcarrier[10:-10], kernel_size=55)  # 尝试窗口大小 3，可根据效果调整
    del x, y
    collected = gc.collect()
    print(f"回收对象数：{collected}")
    return np.concatenate((snr_per_subcarrier[:10], filtered_data, snr_per_subcarrier[-10:]))



def DNCNN_EVALUATE(db_value, model_name, channel_name, DLNET):
    x, y = load_channel_data('D:/AI_Filter/', db_value, 10, 1, [channel_name],
                             '4x80')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检查是否有可用的GPU
    next_x_test = torch.from_numpy(x).float().to(device)
    next_y_test = torch.from_numpy(y).float().to(device)
    next_test_dataset = CustomDataset(next_x_test, next_y_test)
    test_loader = DataLoader(next_test_dataset, batch_size=64, shuffle=False, num_workers=0)
    all_p = []

    checkpoint_save_path = f"../{model_name}/checkpoint/{channel_name}/{model_name}_margin0_num10_epoch200_model{model_name}_setting4x80_dB{snr}_channel{channel_name}.weights.pth"

    DLNET.load_state_dict(torch.load(checkpoint_save_path, weights_only=True))

    # 测试阶段
    DLNET.eval()

    with torch.no_grad():
        for inputs, labels in test_loader:
            batch_p = DLNET(inputs)
            all_p.append(batch_p.cpu().detach())

    p = torch.cat(all_p, dim=0)
    data_len = 8000

    x_test = x.reshape(data_len, 1001, 1)
    y_test = y.reshape(data_len, 1001, 1)
    p_test = p.numpy().reshape(data_len, 1001, 1)
    loss = x_test - y_test

    scidx4x = np.hstack((range(-500, -2, 1), range(3, 501, 1)))
    indices = scidx4x + 500
    get_evm_func_for_mid(indices[:], p_test, x_test, y_test)

    p_test = p.numpy().reshape(8000, 1001)
    x = x.reshape(8000, 1001)


    import gc
    collected = gc.collect()
    print(f"回收对象数：{collected}")

    x = x - p_test
    y = y.reshape(data_len, 1001)
    # 初始化 SNR 数组
    snr_per_subcarrier = np.zeros((1001))

    # 计算每个子载波的 SNR
    for subcarrier in range(1001):
        signal_power_subcarrier = np.mean(np.abs(x[:, subcarrier]) ** 2)
        noise_power_subcarrier = np.mean(
            np.abs(x[:, subcarrier] - y[:, subcarrier]) ** 2)

        snr_per_subcarrier[subcarrier] = 10 * np.log10(signal_power_subcarrier / noise_power_subcarrier)

    from scipy.signal import medfilt
    filtered_data = medfilt(snr_per_subcarrier[10:-10], kernel_size=55)  # 尝试窗口大小 3，可根据效果调整
    del x, y
    collected = gc.collect()
    print(f"回收对象数：{collected}")
    return np.concatenate((snr_per_subcarrier[:10], filtered_data, snr_per_subcarrier[-10:]))


def LSNET_EVALUATE(db_value, model_name, channel_name, DLNET):
    x, y = load_channel_data('D:/AI_Filter/', db_value, 10, 1, [channel_name],
                             '4x80')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检查是否有可用的GPU
    next_x_test = torch.from_numpy(x).float().to(device)
    next_y_test = torch.from_numpy(y).float().to(device)
    next_test_dataset = CustomDataset(next_x_test, next_y_test)
    test_loader = DataLoader(next_test_dataset, batch_size=64, shuffle=False, num_workers=0)
    all_p = []

    checkpoint_save_path = f"../{model_name}/checkpoint/{channel_name}/{model_name}_margin0_num10_epoch200_model{DLNET.name}_setting4x80_dB{db_value}_channel{channel_name}.weights.pth"

    DLNET.load_state_dict(torch.load(checkpoint_save_path, weights_only=True))


    # 测试阶段
    DLNET.eval()

    with torch.no_grad():
        for inputs, labels in test_loader:
            batch_p = DLNET(inputs)
            all_p.append(batch_p.cpu().detach())

    p = torch.cat(all_p, dim=0)
    data_len = 8000

    x_test = x.reshape(data_len, 1001, 1)
    y_test = y.reshape(data_len, 1001, 1)
    p_test = p.numpy().reshape(data_len, 1001, 1)
    loss = x_test - y_test

    scidx4x = np.hstack((range(-500, -2, 1), range(3, 501, 1)))
    indices = scidx4x + 500
    get_evm_func_for_mid(indices[:], p_test, x_test, y_test)

    p_test = p.numpy().reshape(8000, 1001)
    x = x.reshape(8000, 1001)

    import gc
    collected = gc.collect()
    print(f"回收对象数：{collected}")

    x = x - p_test
    y = y.reshape(data_len, 1001)
    # 初始化 SNR 数组
    snr_per_subcarrier = np.zeros((1001))

    # 计算每个子载波的 SNR
    for subcarrier in range(1001):
        signal_power_subcarrier = np.mean(np.abs(x[:, subcarrier]) ** 2)
        noise_power_subcarrier = np.mean(
            np.abs(x[:, subcarrier] - y[:, subcarrier]) ** 2)

        snr_per_subcarrier[subcarrier] = 10 * np.log10(signal_power_subcarrier / noise_power_subcarrier)

    from scipy.signal import medfilt
    filtered_data = medfilt(snr_per_subcarrier[10:-10], kernel_size=55)  # 尝试窗口大小 3，可根据效果调整
    del x, y
    collected = gc.collect()
    print(f"回收对象数：{collected}")
    return np.concatenate((snr_per_subcarrier[:10], filtered_data, snr_per_subcarrier[-10:]))

def plot_func(ans_list, model_list, channel_name, db_value):


    int_colors = [(150, 16, 69), (249, 183, 109), (136, 127, 216), (73, 101, 175), (128, 203, 164)]
    # 将整数 RGB 颜色转换为 0 到 1 之间的浮点数
    colors = [(r / 255, g / 255, b / 255) for r, g, b in int_colors]


    # 绘制 SNR 随子载波变化的图像
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 15,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'legend.fontsize': 24,  # 增大图例字体到14
        'legend.title_fontsize': 24
    })

    for i, item in enumerate(zip(ans_list, model_list)):
        plt.plot(range(0, 1001), item[0], label=item[1], color=colors[i],
                 linewidth=1.5,  # 增大线宽
                 )

    plt.xlabel('Sub-carrier Index',fontsize=24)
    # plt.xticks(rotation=45)
    plt.ylabel(f'The SNR of Sub-carriers (dB) in {channel_name}', fontsize=24)
    # plt.grid(True)
    # 增强图例可读性
    plt.legend(loc='best',
               framealpha=0.9,
               edgecolor='black',  # 添加边框
               handlelength=1.5,  # 增大图例线条长度
               handletextpad=0.5,  # 增大文本与标记间距
               borderpad=0.8)  # 增大图例内边距


    plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检查是否有可用的GPU

    channel_name = 'ChD'
    # snr_list = [0, 10, 20, 30, 40]
    snr_list = [ 40, ]
    model_list = ["GLSNet", "LSNet", "DNCNN", "WIENER FILTER"]

    for snr in tqdm(snr_list, desc="Processing sizes", unit="size"):
        DLNET = LSGhostNet_1D().to(device)
        ans1 = GLSNET_EVALUATE(snr, 'GHOST_LSNET', channel_name, DLNET)

        DLNET = LSNet_1D().to(device)
        ans2 = LSNET_EVALUATE(snr,  'LSNET', channel_name, DLNET)
        #
        DLNET = DNCNN().to(device)
        ans3 = DNCNN_EVALUATE(snr, 'DNCNN', channel_name, DLNET)

        ans4 = WIENER_FILTER_LOAD(snr, channel_name)
        ans_list = [ans1, ans2, ans3, ans4]
        #
        plot_func(ans_list, model_list, channel_name, snr)

