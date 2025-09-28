import numpy as np
from scipy.signal import wiener
import matplotlib.pyplot as plt

from LSNET.LSNet import load_channel_data
from GHOST_LSNET.Ghost_LSNet import get_evm_func_for_mid



def get_data_of_GLS(dbvalue, channel_name):
    import torch
    import torch.nn as nn

    from LSNET.LSNet import load_channel_data, CustomDataset, DataLoader, LSNet_1D
    from GHOST_LSNET.Ghost_LSNet import LSGhostNet_1D, get_evm_func_for_mid

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

    x, y = load_channel_data('D:/AI_Filter/', dbvalue, 10, 1, [channel_name],
                             '4x80')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检查是否有可用的GPU
    next_x_test = torch.from_numpy(x).float().to(device)
    next_y_test = torch.from_numpy(y).float().to(device)
    next_test_dataset = CustomDataset(next_x_test, next_y_test)
    test_loader = DataLoader(next_test_dataset, batch_size=64, shuffle=False, num_workers=0)
    all_p = []
    DLNET = LSGhostNet_1D().to(device)

    checkpoint_save_path = f"../GHOST_LSNET/checkpoint/{channel_name}/GHOST_LSNET_margin10_num10_epoch200_model{DLNET.name}_setting4x80_dB{dbvalue}_channel{channel_name}.weights.pth"

    DLNET.load_state_dict(torch.load(checkpoint_save_path, weights_only=True))

    mode = 'MLP'

    # 测试阶段
    DLNET.eval()



    checkpoint_save_path1 = f"../GHOST_LSNET/checkpoint/{channel_name}/num10epoch2000_Baseline_MLP_4x80_{dbvalue}dB_{channel_name}_margin10.weights.pth"
    checkpoint_save_path2 = f"../GHOST_LSNET/checkpoint/{channel_name}/num10epoch2000_Baseline_MLP_4x80_{dbvalue}dB_{channel_name}_margin-10.weights.pth"
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
    p_test = p.numpy().reshape(data_len, 1001, 1)
    filtered = x_test[0] - p_test[0]
    return filtered.reshape(1001), x_test[0].reshape(1001)

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


    y_test = np.apply_along_axis(lambda x: wiener(x, mysize=optsize), 1, tnoisyChannel)

    x_test = x.reshape(num_samples, 1001, 1)
    y_test = y_test.reshape(num_samples, 1001, 1)
    y = y.reshape(num_samples, 1001, 1)

    scidx4x = np.hstack((range(-500, -2, 1), range(3, 501, 1)))
    indices = scidx4x + 500

    evm = get_evm_func_for_mid(indices[:],x_test-y_test, x_test, y)
    return mmse_opt, evm


def plot_loss_fig(dbvalue, channel_name):
    x, y = load_channel_data("D:/AI_Filter/", dbvalue, 10, 1, [channel_name], '4x80')
    x = x[0]
    y = y[0]

    tperfectChannel = y.reshape(1001)
    tnoisyChannel = x.reshape(1001)

    mmse_opt = calculate_mse(tperfectChannel, tnoisyChannel)
    print("mmse_opt", mmse_opt)
    optsize = 0

    for size in range(3, 20):
        # 应用维纳滤波，对每一条数据单独进行滤波
        print("size test:", size)
        filtered_signal = np.apply_along_axis(lambda x: wiener(x, mysize=size), 0, tnoisyChannel)
        mmse = calculate_mse(tperfectChannel, filtered_signal)
        if mmse < mmse_opt:
            mmse_opt = mmse
            optsize = size
            print("mmse", mmse, "mmse_opt", mmse_opt, "optsize", optsize)

    y_test = np.apply_along_axis(lambda x: wiener(x, mysize=optsize), 0, tnoisyChannel)

    GLS_FILTERED, origin = get_data_of_GLS(dbvalue, channel_name)
    DNCNN_FILTERED, _ = get_data_of_DNCNN(dbvalue, channel_name)

    # print(tnoisyChannel)
    # print(tperfectChannel)
    # print(y_test)
    # 创建图形
    plt.figure(figsize=(10, 6))
    receivedsignal = tnoisyChannel
    filtered = y_test

    x = np.arange(0, 1001)
    lft = 120
    keeplen = 150
    # 绘制原始信号折线
    plt.plot(x[lft:keeplen], receivedsignal[lft:keeplen], label='received signal', color='grey', marker='o')
    # 绘制滤波后信号折线
    plt.plot(x[lft:keeplen], filtered[lft:keeplen], label='filtered signal', color='orange', marker='s')
    plt.plot(x[lft:keeplen], GLS_FILTERED[lft:keeplen], label='predicted signal of GLS', color='green', marker='p')
    plt.plot(x[lft:keeplen], tperfectChannel[lft:keeplen], label='ideal signal', color='red', marker='^',linestyle=':')
    plt.plot(x[lft:keeplen], DNCNN_FILTERED[lft:keeplen], label='predicted signal of DNCNN', color='blue', marker='+')


    # # 绘制失真点
    # plt.scatter(x[123], filtered[122], color='black', marker='*', s=200, label='distortion')
    # 标注失真点说明
    plt.text(x[123]+0.8, filtered[123], 'distortion', fontsize=10, ha='left')
    # 绘制一个从 (122, -0.23) 到 (124, -0.23) 的水平箭头
    plt.arrow(x[123], filtered[123], 0.6, 0.0001, color='black', width=0.0001, head_width=0.0005, head_length=0.2)


    # 添加标题和坐标轴标签
    plt.title('Signal Filtering Distortion')
    plt.xlabel('Sub-carrier Index')
    plt.ylabel('Channel Impulse Response')

    # 设置 y 轴范围，让显示更贴合示例图
    # plt.ylim(-0.5, 0.5)
    # 添加图例
    plt.legend()

    # 显示图形
    plt.grid(True)
    plt.show()

import torch.nn as nn

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


'''
input: model to be analysed
output: model calculation flops
'''
def get_flops(model):
    from thop import profile
    import torch

    model = model()
    # 生成随机输入
    input = torch.randn(8, 1, 20)
    # 分析计算量和参数数量
    flops, params = profile(model, inputs=(input,))

    print(f"FLOPs: {flops / 1e9:.6f} GFLOPs, Params: {params / 1e6:.3f} M")





def get_data_of_DNCNN(dbvalue, channel_name):
    import torch
    import torch.nn as nn

    from LSNET.LSNet import load_channel_data, CustomDataset, DataLoader

    from DNCNN.DNCNN import DNCNN



    x, y = load_channel_data('D:/AI_Filter/', dbvalue, 10, 1, [channel_name],
                             '4x80')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检查是否有可用的GPU
    next_x_test = torch.from_numpy(x).float().to(device)
    next_y_test = torch.from_numpy(y).float().to(device)
    next_test_dataset = CustomDataset(next_x_test, next_y_test)
    test_loader = DataLoader(next_test_dataset, batch_size=64, shuffle=False, num_workers=0)
    all_p = []
    DLNET = DNCNN().to(device)

    checkpoint_save_path = f"../DNCNN/checkpoint/{channel_name}/DNCNN_margin0_num10_epoch200_modelDNCNN_setting4x80_dB{dbvalue}_channel{channel_name}.weights.pth"

    DLNET.load_state_dict(torch.load(checkpoint_save_path, weights_only=True))


    # 测试阶段
    DLNET.eval()


    with torch.no_grad():
        for inputs, labels in test_loader:
            output = DLNET(inputs)
            all_p.append(output.cpu().detach())

    p = torch.cat(all_p, dim=0)
    data_len = 8000

    x_test = x.reshape(data_len, 1001, 1)
    p_test = p.numpy().reshape(data_len, 1001, 1)
    filtered = x_test[0] - p_test[0]
    return filtered.reshape(1001), x_test[0].reshape(1001)

if __name__ == "__main__":
    from LSNET.LSNet import LSNet_1D
    from GHOST_LSNET.Ghost_LSNet import LSGhostNet_1D
    from plot_subcarriers import Baseline_MLP
    get_flops(Baseline_MLP)
    # get_flops(LSNet_1D)

    # plot_loss_fig(20, 'ChB')