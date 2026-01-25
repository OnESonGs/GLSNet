import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn

from LSNET.LSNet import load_channel_data, CustomDataset, DataLoader, LSNet_1D
from GHOST_LSNET.Ghost_LSNet import LSGhostNet_1D
from DNCNN.DNCNN import DNCNN
from utils import WIENER_FILTER_LOAD, mean_confidence_interval

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


def load_model_GLSNET(db_value, channel_name, model_name, DLNET, XX, YY, loop):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_save_path = f"./{model_name}/checkpoint/{channel_name}/{model_name}_margin10_num10_epoch200_model{DLNET.name}_setting4x80_dB{db_value}_channel{channel_name}.weights.pth"
    DLNET.load_state_dict(torch.load(checkpoint_save_path, weights_only=True, map_location=device))

    DLNET.eval()
    checkpoint_save_path1 = f"./{model_name}/checkpoint/{channel_name}/num10epoch2000_Baseline_MLP_4x80_{db_value}dB_{channel_name}_margin10.weights.pth"
    checkpoint_save_path2 = f"./{model_name}/checkpoint/{channel_name}/num10epoch2000_Baseline_MLP_4x80_{db_value}dB_{channel_name}_margin-10.weights.pth"
    mlp1 = Baseline_MLP().to(device)
    mlp2 = Baseline_MLP().to(device)
    mlp1.load_state_dict(torch.load(checkpoint_save_path1, weights_only=True, map_location=device))
    mlp2.load_state_dict(torch.load(checkpoint_save_path2, weights_only=True, map_location=device))

    mlp1.eval()
    mlp2.eval()

    mse_list = []
    for idx in range(loop):
        indices = np.random.choice(8000, size=1000, replace=True)
        x = XX[indices]
        y = YY[indices]

        next_x_test = torch.from_numpy(x).float().to(device)
        next_y_test = torch.from_numpy(y).float().to(device)
        next_test_dataset = CustomDataset(next_x_test, next_y_test)
        test_loader = DataLoader(next_test_dataset, batch_size=64, shuffle=False, num_workers=0)
        all_p = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                output1 = mlp1(inputs[:, :, :20])
                output2 = DLNET(inputs)
                output3 = mlp2(inputs[:, :, -20:])
                batch_p = torch.cat((output1, output2[:, :, 10:-10], output3), dim=2)
                all_p.append(batch_p.cpu().detach())

        p = torch.cat(all_p, dim=0)

        x_test = x.reshape(-1, 1001, 1)
        y_test = y.reshape(-1, 1001, 1)
        p_test = p.numpy().reshape(-1, 1001, 1)

        mse = calculate_mse(y_test, x_test-p_test)
        mse_list.append(mse)
        import gc
        collected = gc.collect()
        print(f"COLLECTION NUMBER: {collected}")

    return mse_list



def load_model_LSNET(db_value, channel_name, model_name, DLNET, XX, YY, loop):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_save_path = f"./{model_name}/checkpoint/{channel_name}/{model_name}_margin0_num10_epoch200_model{DLNET.name}_setting4x80_dB{db_value}_channel{channel_name}.weights.pth"
    DLNET.load_state_dict(torch.load(checkpoint_save_path, weights_only=True, map_location=device))
    DLNET.eval()

    mse_list = []
    for idx in range(loop):
        indices = np.random.choice(8000, size=1000, replace=True)
        x = XX[indices]
        y = YY[indices]

        next_x_test = torch.from_numpy(x).float().to(device)
        next_y_test = torch.from_numpy(y).float().to(device)
        next_test_dataset = CustomDataset(next_x_test, next_y_test)
        test_loader = DataLoader(next_test_dataset, batch_size=64, shuffle=False, num_workers=0)
        all_p = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                batch_p = DLNET(inputs)
                all_p.append(batch_p.cpu().detach())

        p = torch.cat(all_p, dim=0)
        x_test = x.reshape(-1, 1001, 1)
        y_test = y.reshape(-1, 1001, 1)
        p_test = p.numpy().reshape(-1, 1001, 1)

        mse = calculate_mse(y_test, x_test-p_test)
        mse_list.append(mse)
        import gc
        collected = gc.collect()
        print(f"COLLECTION NUMBER: {collected}")
    return mse_list

def load_model_DNCNN(db_value, model_name, channel_name, DLNET, XX, YY, loop):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_save_path = f"./{model_name}/checkpoint/{channel_name}/{model_name}_margin0_num10_epoch200_model{model_name}_setting4x80_dB{db_value}_channel{channel_name}.weights.pth"
    DLNET.load_state_dict(torch.load(checkpoint_save_path, weights_only=True, map_location=device))
    DLNET.eval()

    mse_list = []
    for idx in range(loop):
        indices = np.random.choice(8000, size=1000, replace=True)
        x = XX[indices]
        y = YY[indices]

        next_x_test = torch.from_numpy(x).float().to(device)
        next_y_test = torch.from_numpy(y).float().to(device)
        next_test_dataset = CustomDataset(next_x_test, next_y_test)
        test_loader = DataLoader(next_test_dataset, batch_size=64, shuffle=False, num_workers=0)
        all_p = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                batch_p = DLNET(inputs)
                all_p.append(batch_p.cpu().detach())

        p = torch.cat(all_p, dim=0)

        x_test = x.reshape(-1, 1001, 1)
        y_test = y.reshape(-1, 1001, 1)
        p_test = p.numpy().reshape(-1, 1001, 1)

        mse = calculate_mse(y_test, x_test-p_test)
        mse_list.append(mse)
        import gc
        collected = gc.collect()
        print(f"COLLECTION NUMBER: {collected}")
    return mse_list


def calculate_mse(original, filtered):
    return np.mean((original - filtered) ** 2)

def plotfunc_mse(arr_list, model_list, channel_name):
    x_axis = np.array([0, 10, 20, 30, 40])
    int_colors = [(150, 16, 69), (249, 183, 109), (136, 127, 216), (73, 101, 175), (128, 203, 164), (49, 83, 109),
                  (128, 101, 109)]
    main_colors = [(r/255, g/255, b/255) for r, g, b in int_colors]

    marker_list = ['o', '^', 'p', 's', '+', 'd', '*']
    plt.figure(figsize=(12, 7), dpi=300)
    ax = plt.gca()
    ax.set_axisbelow(True)

    for idx, (mse_data, model_name) in enumerate(zip(arr_list, model_list)):
        mse_mean = mse_data[:, 0]
        mse_low = mse_data[:, 1]
        mse_high = mse_data[:, 2]

        main_color = main_colors[idx % len(main_colors)]

        ax.plot(x_axis, mse_high, linestyle='-.', color=main_color, linewidth=2.0)
        ax.plot(x_axis, mse_low, linestyle='-.', color=main_color, linewidth=2.0)
        ax.plot(x_axis, mse_mean, linestyle='-', marker=marker_list[idx], markersize=12, label=f'{model_name} (Mean ± 99% CI)', color=main_color, linewidth=1.0)

    plt.xlabel(f'SNR of Original Signal in {channel_name}', fontsize=18)
    plt.ylabel(f'MSE of Filtered Signal in {channel_name}', fontsize=18)
    plt.yscale('log')
    plt.tick_params(axis='both', which='major', labelsize=18)

    # plt.yticks(yticks)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.legend()
    plt.legend(loc='upper right', fontsize=18, framealpha=1, edgecolor='black')
    plt.tight_layout()

    # plt.show()
    plt.savefig(
        f'{channel_name}_MSE.eps',
        format='eps',
        dpi=300,
        bbox_inches='tight',
        backend='ps'
    )
    plt.close()



def get_mse(channel_name, loop, device):
    snr_list = [0, 10, 20, 30, 40]
    model_list = ["GHOST_LSNET", "LSNET", "DNCNN", "WIENER FILTER"]

    arr1 = []
    arr2 = []
    arr3 = []
    arr4 = []
    for snr in tqdm(snr_list, desc="Processing sizes", unit="size"):
        XX, YY = load_channel_data('../', snr, 10, 1, [channel_name],
                                   '4x80')

        DLNET = LSGhostNet_1D().to(device)
        mse = np.array(load_model_GLSNET(snr, channel_name, 'GHOST_LSNET', DLNET, XX, YY, loop))
        arr1.append(mean_confidence_interval(mse))

        DLNET = LSNet_1D().to(device)
        mse = np.array(load_model_LSNET(snr, channel_name, 'LSNET', DLNET, XX, YY, loop))
        arr2.append(mean_confidence_interval(mse))

        DLNET = DNCNN().to(device)
        mse = np.array(load_model_DNCNN(snr, 'DNCNN', channel_name, DLNET, XX, YY, loop))
        arr3.append(mean_confidence_interval(mse))

        mse = np.array(WIENER_FILTER_LOAD(snr, channel_name, XX, YY, loop))
        arr4.append(mean_confidence_interval(mse))

    arr_list = np.array([arr1, arr2, arr3, arr4])
    plotfunc_mse(arr_list, model_list, channel_name)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    get_mse(channel_name='ChB', loop=1000, device=device)
    get_mse(channel_name='ChD', loop=1000, device=device)


