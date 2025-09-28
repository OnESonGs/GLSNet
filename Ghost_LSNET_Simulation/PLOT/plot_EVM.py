import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # 新增进度条库

import torch
import torch.nn as nn

from LSNET.LSNet import load_channel_data, CustomDataset, DataLoader, LSNet_1D
from GHOST_LSNET.Ghost_LSNet import LSGhostNet_1D, get_evm_func_for_mid
from utils import WIENER_FILTER_LOAD
from DNCNN.DNCNN import DNCNN

# 定义边带baselin模型(MLP)
class Baseline_MLP(nn.Module):
    def __init__(self):
        super(Baseline_MLP, self).__init__()
        self.name = 'Baseline_MLP'
        self.fc1 = nn.Linear(20, 16)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(16, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        y = self.fc3(x)
        return y


def load_model_GLSNET(db_value, channel_name, model_name, DLNET):
    x, y = load_channel_data('D:/AI_Filter/', db_value, 10, 1, [channel_name],
                             '4x80')
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

    scidx4x = np.hstack((range(-500, -2, 1), range(3, 501, 1)))
    indices = scidx4x + 500

    evm = get_evm_func_for_mid(indices[:], p_test, x_test, y_test)
    import gc
    collected = gc.collect()
    print(f"回收对象数：{collected}")
    return y_test, x_test-p_test, evm


def load_model_LSNET_MID(db_value, channel_name, model_name, DLNET):
    x, y = load_channel_data('D:/AI_Filter/', db_value, 10, 1, [channel_name],
                             '4x80')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检查是否有可用的GPU
    next_x_test = torch.from_numpy(x).float().to(device)
    next_y_test = torch.from_numpy(y).float().to(device)
    next_test_dataset = CustomDataset(next_x_test, next_y_test)
    test_loader = DataLoader(next_test_dataset, batch_size=64, shuffle=False, num_workers=0)
    all_p = []

    checkpoint_save_path = f"../{model_name}/checkpoint/{channel_name}/{model_name}_margin10_num10_epoch200_model{DLNET.name}_setting4x80_dB{db_value}_channel{channel_name}.weights.pth"

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

    scidx4x = np.hstack((range(-500, -2, 1), range(3, 501, 1)))
    indices = scidx4x + 500

    evm = get_evm_func_for_mid(indices[10:-10], p_test, x_test, y_test)
    import gc
    collected = gc.collect()
    print(f"回收对象数：{collected}")
    return y_test[:,10:-10,:], x_test[:,10:-10,:]-p_test[:,10:-10,:], evm



def load_model_LSNET_ALL(db_value, channel_name, model_name, DLNET):
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

    scidx4x = np.hstack((range(-500, -2, 1), range(3, 501, 1)))
    indices = scidx4x + 500

    evm = get_evm_func_for_mid(indices[:], p_test, x_test, y_test)
    import gc
    collected = gc.collect()
    print(f"回收对象数：{collected}")
    return y_test, x_test-p_test, evm

def load_model_DNCNN(db_value, model_name, channel_name, DLNET):
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

    evm = get_evm_func_for_mid(indices[:], p_test, x_test, y_test)
    import gc
    collected = gc.collect()
    print(f"回收对象数：{collected}")
    return y_test, x_test-p_test, evm


# 计算均方误差（MSE）
def calculate_mse(original, filtered):
    return np.mean((original - filtered) ** 2)


def plotfunc_mse(arr_list, model_list, channel_name):
    # 定义 SNR 值
    snr = np.array([0, 10, 20, 30, 40])

    int_colors = [(150, 16, 69), (249, 183, 109), (136, 127, 216), (73, 101, 175), (128, 203, 164), (49, 83, 109),
                  (128, 101, 109)]
    # 将整数 RGB 颜色转换为 0 到 1 之间的浮点数
    colors = [(r / 255, g / 255, b / 255) for r, g, b in int_colors]

    marker_list = ['o', '*', '^', 's', 'p', 'd', '+']

    # 创建图形并设置全局字体大小
    plt.figure(figsize=(8, 6))
    plt.rcParams.update({
        'font.size': 14,  # 全局字体大小
        'axes.titlesize': 16,  # 轴标题大小
        'axes.labelsize': 15,  # 轴标签大小
        'xtick.labelsize': 24,  # X轴刻度标签大小
        'ytick.labelsize': 24,  # Y轴刻度标签大小
        'legend.fontsize': 24,  # 增大图例字体到14
        'legend.title_fontsize': 24
    })

    # 绘制图形
    for i, item in enumerate(zip(arr_list, model_list)):
        plt.plot(snr, item[0],
                 marker=marker_list[i],
                 markersize=15,  # 增大标记尺寸
                 linewidth=1.5,  # 增大线宽
                 label=item[1],
                 color=colors[i])

    plt.xlabel(f'SNR of Original Signal in {channel_name}', fontsize=24)
    plt.ylabel(f'MSE of Filtered Signal in {channel_name}', fontsize=24)
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # 增强图例可读性
    plt.legend(loc='best',
               # ncol=2,  # 关键修改：设置图例列数为2
               framealpha=0.9,
               edgecolor='black',  # 添加边框
               handlelength=1.5,  # 增大图例线条长度
               handletextpad=0.5,  # 增大文本与标记间距
               borderpad=0.8     # 增大图例内边距
               )

    # # 在绘图代码末尾添加范围控制（示例）
    # plt.xlim(0, 40)  # X轴固定显示0-40 SNR值
    # plt.ylim(1e-6, 1e-2)  # Y轴对数范围：1e-6 到 1e-2

    # 确保布局紧凑
    plt.tight_layout()
    plt.show()


def plotfunc_snr(arr_list, model_list, channel_name):
    # 定义 SNR 值
    snr = np.array([0, 10, 20, 30, 40])

    int_colors = [(150, 16, 69), (249, 183, 109), (136, 127, 216), (73, 101, 175), (128, 203, 164), (49, 83, 109),
                  (128, 101, 109)]
    # 将整数 RGB 颜色转换为 0 到 1 之间的浮点数
    colors = [(r / 255, g / 255, b / 255) for r, g, b in int_colors]

    marker_list = ['o', '*', '^', 's', 'p', 'd', '+']

    # 创建图形并设置全局字体大小
    plt.figure(figsize=(8, 6))
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 15,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'legend.fontsize': 24,  # 增大图例字体到14
        'legend.title_fontsize': 24
    })

    # 绘制图形
    for i, item in enumerate(zip(arr_list, model_list)):
        plt.plot(snr, item[0],
                 marker=marker_list[i],
                 markersize=15,  # 增大标记尺寸
                 linewidth=1.5,  # 增大线宽
                 label=item[1],
                 color=colors[i])

    plt.xlabel(f'SNR of Original Signal in {channel_name}', fontsize=24)
    plt.ylabel(f'SNR Enhancement of Filtered Signal in {channel_name}', fontsize=24)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # 增强图例可读性
    plt.legend(loc='best',
               framealpha=0.9,
               edgecolor='black',  # 添加边框
               handlelength=1.5,  # 增大图例线条长度
               handletextpad=0.5,  # 增大文本与标记间距
               borderpad=0.8)  # 增大图例内边距

    # 确保布局紧凑
    plt.tight_layout()
    plt.show()








if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检查是否有可用的GPU

    channel_name = 'ChB'
    snr_list = [0, 10, 20, 30, 40]
    model_list = ["GLSNet", "LSNet", "DNCNN", "WIENER FILTER", "LSNet_Stu", "GLSNet_Stu", "DNCNN_Stu"]
    DLNET = LSGhostNet_1D().to(device)
    arr1 = []
    evm_arr1 =[]
    for snr in tqdm(snr_list, desc="Processing sizes", unit="size"):
        x, y, evm = load_model_GLSNET(snr, channel_name, 'GHOST_LSNET', DLNET)
        mse = calculate_mse(x, y)
        arr1.append(mse)
        evm_arr1.append(evm)

    DLNET = LSNet_1D().to(device)
    arr2 = []
    evm_arr2 =[]

    for snr in tqdm(snr_list, desc="Processing sizes", unit="size"):
        x, y, evm = load_model_LSNET_ALL(snr, channel_name, 'LSNET', DLNET)
        mse = calculate_mse(x, y)
        arr2.append(mse)
        evm_arr2.append(evm)


    DLNET = DNCNN().to(device)
    arr3 = []
    evm_arr3 =[]
    for snr in tqdm(snr_list, desc="Processing sizes", unit="size"):
        x, y, evm = load_model_DNCNN(snr, 'DNCNN', channel_name, DLNET)
        mse = calculate_mse(x, y)
        arr3.append(mse)
        evm_arr3.append(evm)

    arr4 = []
    evm_arr4 =[]
    for snr in tqdm(snr_list, desc="Processing sizes", unit="size"):
        mse, evm = WIENER_FILTER_LOAD(snr, channel_name)
        arr4.append(mse)
        evm_arr4.append(evm)

    arr_list = [arr1, arr2, arr3, arr4]
    snr_list = [evm_arr1, evm_arr2, evm_arr3, evm_arr4]

    plotfunc_mse(arr_list, model_list, channel_name)


