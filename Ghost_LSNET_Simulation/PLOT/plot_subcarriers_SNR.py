import gc
from scipy.signal import wiener
from scipy.signal import medfilt

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from LSNET.LSNet import load_channel_data, CustomDataset, DataLoader, LSNet_1D
from GHOST_LSNET.Ghost_LSNet import LSGhostNet_1D, get_evm_func_for_mid
from DNCNN.DNCNN import DNCNN
from plot_MSE import  Baseline_MLP
from utils import mean_confidence_interval

def calculate_mse(original, filtered):
    return np.mean((original - filtered) ** 2)

def WIENER_FILTER_LOAD(dbvalue, channel_name, X, Y, loop):
    snr_list = []
    for idx in range(loop):
        indices = np.random.choice(8000, size=1000, replace=True)
        x = X[indices]
        y = Y[indices]

        num_signals = 1001
        tperfectChannel = y.reshape(-1, num_signals)
        tnoisyChannel = x.reshape(-1, num_signals)

        mmse_opt = calculate_mse(tperfectChannel, tnoisyChannel)
        # print("mmse_opt", mmse_opt)
        optsize = 0

        for size in range(3,20):
            # print("size test:", size)
            filtered_signal = np.apply_along_axis(lambda x: wiener(x, mysize=size), 1, tnoisyChannel)
            mmse = calculate_mse(tperfectChannel, filtered_signal)
            if mmse < mmse_opt:
                mmse_opt = mmse
                optsize = size
                # print("mmse", mmse, "mmse_opt", mmse_opt, "optsize", optsize)

        y = np.apply_along_axis(lambda x: wiener(x, mysize=optsize), 1, tnoisyChannel)
        x = tnoisyChannel

        snr_per_subcarrier = np.zeros((1001))

        for subcarrier in range(1001):
            signal_power_subcarrier = np.mean(np.abs(x[:, subcarrier]) ** 2)
            noise_power_subcarrier = np.mean(
                np.abs(x[:, subcarrier] - y[:, subcarrier]) ** 2)

            snr_per_subcarrier[subcarrier] = 10 * np.log10(signal_power_subcarrier / noise_power_subcarrier)

        filtered_data = medfilt(snr_per_subcarrier[10:-10], kernel_size=55)
        del x, y
        collected = gc.collect()
        print(f"COLLECTION NUMBER: {collected}")
        snr_list.append(np.concatenate((snr_per_subcarrier[:10], filtered_data, snr_per_subcarrier[-10:])))
    return snr_list


def GLSNET_EVALUATE(db_value, model_name, channel_name, DLNET, X, Y, loop):

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

    snr_list = []
    for idx in range(loop):
        indices = np.random.choice(8000, size=1000, replace=True)
        x = X[indices]
        y = Y[indices]
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

        scidx4x = np.hstack((range(-500, -2, 1), range(3, 501, 1)))
        indices = scidx4x + 500
        get_evm_func_for_mid(indices[:], p_test, x_test, y_test)

        p_test = p.numpy().reshape(-1, 1001)
        x = x.reshape(-1, 1001)

        collected = gc.collect()
        print(f"COLLECTION NUMBER: {collected}")

        x = x - p_test
        y = y.reshape(-1, 1001)
        snr_per_subcarrier = np.zeros((1001))

        for subcarrier in range(1001):
            signal_power_subcarrier = np.mean(np.abs(x[:, subcarrier]) ** 2)
            noise_power_subcarrier = np.mean(
                np.abs(x[:, subcarrier]-y[:, subcarrier]) ** 2)

            snr_per_subcarrier[subcarrier] = 10 * np.log10(signal_power_subcarrier / noise_power_subcarrier)

        filtered_data = medfilt(snr_per_subcarrier[10:-10], kernel_size=55)
        del x, y
        collected = gc.collect()
        print(f"COLLECTION NUMBER: {collected}")

        snr_list.append(np.concatenate((snr_per_subcarrier[:10], filtered_data, snr_per_subcarrier[-10:])))
    return snr_list



def DNCNN_EVALUATE(db_value, model_name, channel_name, DLNET, X, Y, loop):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_save_path = f"./{model_name}/checkpoint/{channel_name}/{model_name}_margin0_num10_epoch200_model{model_name}_setting4x80_dB{db_value}_channel{channel_name}.weights.pth"
    DLNET.load_state_dict(torch.load(checkpoint_save_path, weights_only=True, map_location=device))
    DLNET.eval()

    snr_list = []
    for idx in range(loop):
        indices = np.random.choice(8000, size=1000, replace=True)
        x = X[indices]
        y = Y[indices]

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

        scidx4x = np.hstack((range(-500, -2, 1), range(3, 501, 1)))
        indices = scidx4x + 500
        get_evm_func_for_mid(indices[:], p_test, x_test, y_test)

        p_test = p.numpy().reshape(-1, 1001)
        x = x.reshape(-1, 1001)

        collected = gc.collect()
        print(f"COLLECTION NUMBER: {collected}")

        x = x - p_test
        y = y.reshape(-1, 1001)
        snr_per_subcarrier = np.zeros((1001))

        for subcarrier in range(1001):
            signal_power_subcarrier = np.mean(np.abs(x[:, subcarrier]) ** 2)
            noise_power_subcarrier = np.mean(
                np.abs(x[:, subcarrier] - y[:, subcarrier]) ** 2)

            snr_per_subcarrier[subcarrier] = 10 * np.log10(signal_power_subcarrier / noise_power_subcarrier)

        from scipy.signal import medfilt
        filtered_data = medfilt(snr_per_subcarrier[10:-10], kernel_size=55)
        del x, y
        collected = gc.collect()
        print(f"COLLECTION NUMBER: {collected}")

        snr_list.append(np.concatenate((snr_per_subcarrier[:10], filtered_data, snr_per_subcarrier[-10:])))
    return snr_list


def LSNET_EVALUATE(db_value, model_name, channel_name, DLNET, X, Y, loop):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_save_path = f"./{model_name}/checkpoint/{channel_name}/{model_name}_margin0_num10_epoch200_model{DLNET.name}_setting4x80_dB{db_value}_channel{channel_name}.weights.pth"

    DLNET.load_state_dict(torch.load(checkpoint_save_path, weights_only=True, map_location=device))
    DLNET.eval()

    snr_list = []
    for idex in range(loop):
        indices = np.random.choice(8000, size=1000, replace=True)
        x = X[indices]
        y = Y[indices]

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

        scidx4x = np.hstack((range(-500, -2, 1), range(3, 501, 1)))
        indices = scidx4x + 500
        get_evm_func_for_mid(indices[:], p_test, x_test, y_test)

        p_test = p.numpy().reshape(-1, 1001)
        x = x.reshape(-1, 1001)

        import gc
        collected = gc.collect()
        print(f"COLLECTION NUMBER: {collected}")

        x = x - p_test
        y = y.reshape(-1, 1001)
        snr_per_subcarrier = np.zeros((1001))

        for subcarrier in range(1001):
            signal_power_subcarrier = np.mean(np.abs(x[:, subcarrier]) ** 2)
            noise_power_subcarrier = np.mean(
                np.abs(x[:, subcarrier] - y[:, subcarrier]) ** 2)

            snr_per_subcarrier[subcarrier] = 10 * np.log10(signal_power_subcarrier / noise_power_subcarrier)

        filtered_data = medfilt(snr_per_subcarrier[10:-10], kernel_size=55)
        del x, y
        collected = gc.collect()
        print(f"COLLECTION NUMBER: {collected}")

        snr_list.append(np.concatenate((snr_per_subcarrier[:10], filtered_data, snr_per_subcarrier[-10:])))
    return snr_list

def plot_func(ans_list, model_list, channel_name, db_value):
    main_int_colors = [(150, 16, 69), (249, 183, 109), (136, 127, 216), (73, 101, 175), (128, 203, 164)]
    main_colors = [(r/255, g/255, b/255) for r, g, b in main_int_colors]
    line_styles = ['-', '--', '-.', ':']

    plt.figure(figsize=(12, 7), dpi=300)
    ax = plt.gca()
    ax.set_axisbelow(True)

    x_axis = np.arange(0, 1001)
    for idx, (snr_data, model_name) in enumerate(zip(ans_list, model_list)):
        snr_mean = snr_data[:, 0]
        snr_low = snr_data[:, 1]
        snr_high = snr_data[:, 2]

        main_color = main_colors[idx % len(main_colors)]
        line_style = line_styles[idx % len(line_styles)]

        ax.plot(
            x_axis, snr_high, linestyle=line_style,
            color=main_color,
            linewidth=3.0  # , marker='.', markersize=2
        )

        ax.plot(
            x_axis, snr_low, linestyle=line_style,
            color=main_color,
            linewidth=3.0  # , marker='.', markersize=2
        )

        ax.plot(
            x_axis, snr_mean, linestyle=line_style,
            color=main_color, label=f'{model_name} (Mean ± 99% CI)',
            linewidth=1.0 # , marker='.', markersize=2
        )

    ax.set_xlabel('Sub-carrier Index', fontsize=18, fontweight='bold')
    ax.set_ylabel(f'The SNR of Sub-carriers (dB) in {channel_name}', fontsize=18, fontweight='bold')
    ax.set_xticks(np.arange(0, 1001, 100))
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='#E0E0E0')
    # ax.legend(loc='upper right', fontsize=10, framealpha=1, edgecolor='black')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.4), fontsize=18, framealpha=1, edgecolor='black')
    plt.tight_layout()

    plt.savefig(
        f'{channel_name}_SNR_per_subcarriers_{db_value}dB.eps',
        format='eps',
        dpi=300,
        bbox_inches='tight',
        backend='ps',
        pad_inches=0.1
    )
    plt.close()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    channel_name = input('Enter channel name: ChB or ChD? \n')
    snr_list = [40, 30, 20, 10, 0]
    model_list = ["GHOST_LSNET", "LSNET", "DNCNN", "WIENER FILTER"]
    loop = 1000

    for snr in tqdm(snr_list, desc="Processing sizes", unit="size"):
        x, y = load_channel_data('../', snr, 10, 1, [channel_name],
                                 '4x80')

        DLNET = LSGhostNet_1D().to(device)
        ans1 = np.array(GLSNET_EVALUATE(snr, 'GHOST_LSNET', channel_name, DLNET, x, y, loop))

        DLNET = LSNet_1D().to(device)
        ans2 = np.array(LSNET_EVALUATE(snr,  'LSNET', channel_name, DLNET, x, y, loop))

        DLNET = DNCNN().to(device)
        ans3 = np.array(DNCNN_EVALUATE(snr, 'DNCNN', channel_name, DLNET, x, y, loop))

        ans4 = np.array(WIENER_FILTER_LOAD(snr, channel_name, x, y, loop))

        arr1 = []
        arr2 = []
        arr3 = []
        arr4 = []
        for idx in range(1001):
            arr1.append(mean_confidence_interval(ans1[:,idx]))
            arr2.append(mean_confidence_interval(ans2[:,idx]))
            arr3.append(mean_confidence_interval(ans3[:,idx]))
            arr4.append(mean_confidence_interval(ans4[:,idx]))

        ans_list = np.array([arr1, arr2, arr3, arr4])
        plot_func(ans_list, model_list, channel_name, snr)

