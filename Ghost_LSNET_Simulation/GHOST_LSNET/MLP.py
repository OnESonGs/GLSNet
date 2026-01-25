import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.data import DataLoader

import torch
import numpy as np
import time
import logging
from pathlib import Path
from torch.optim import Adam

#######################NEURAL NETWORK###############################
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

# LOSS FUNCTION
def my_loss(y_p, y_true):
    return torch.max(torch.abs(y_p - y_true), dim=-1)[0].mean()

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x # noisy data
        self.y = y # perfect data

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.x[idx] - self.y[idx] # noise learning


'''
input: snr value, file index, file number, list of channel name, interpolate mode
output: 2x80 or 4x80 interpolated DC sub carriers (datasize=1000*8*file_num*len(channel_list), 1, 1001) 
'''
def load_channel_data(folder_name, db_value, file_index, file_num, channel_list, interpolate_mode):
    '''
    algorithm of 2x80 or 4x80 interpolated DC sub carriers
    '''
    from scipy.io import loadmat
    import scipy.interpolate as si
    # interpolate DC sub carriers
    def my_interpolate(x, y, x_new):
        f = si.interp1d(x, y, kind='linear', fill_value=(y[0], y[-1]), bounds_error=False)
        y_new = f(x_new)
        return y_new

    def xxx(idx4x, idx2x, t_idx4x, channel):
        tchannel = np.zeros((channel.shape[0], 1001))
        for ii in range(0, channel.shape[0]):
            yt = channel[ii, :]
            yt = yt.reshape(996)
            yt2 = my_interpolate(idx4x, yt, idx2x)
            yt3 = my_interpolate(idx2x, yt2, t_idx4x)
            tchannel[ii, :] = yt3.reshape(1, 1001)
        return tchannel

    def yyy(idx4x, t_idx4x, channel):
        tchannel = np.zeros((channel.shape[0], 1001))
        for ii in range(0, channel.shape[0]):
            yt = channel[ii, :]
            yt = yt.reshape(996)
            yt2 = my_interpolate(idx4x, yt, t_idx4x)
            tchannel[ii, :] = yt2.reshape(1, 1001)
        return tchannel

    dict_perfect_channel = {}
    dict_noisy_channel = {}

    data_len = len(channel_list) * 8000 * file_num # the number if single file is 8000

    if file_num == 1:
        ''' single file '''
        for item in channel_list:
            t_perfect_channel = loadmat(f'{folder_name}dataset/{item}/perfectChannel{item}{db_value}dB_{file_index}.mat')['perfectChannel']
            dict_perfect_channel[item] = np.array(t_perfect_channel)

            t_noisy_channel = loadmat(f'{folder_name}dataset/{item}/noisyChannel{item}{db_value}dB_{file_index}.mat')['noisyChannel']
            dict_noisy_channel[item] = np.array(t_noisy_channel)

        final_noisy_channel = np.vstack([dict_noisy_channel[item] for item in channel_list])
        final_perfect_channel = np.vstack([dict_perfect_channel[item] for item in channel_list])

    else:
        file_index_list = [index for index in range(file_index, file_index+file_num)]
        # print("file_index_list", file_index_list)
        all_noisy_channels = []
        all_perfect_channels = []

        ''' file list '''
        for index in file_index_list:
            for item in channel_list:
                t_perfect_channel = loadmat(f'{folder_name}dataset/{item}/perfectChannel{item}{db_value}dB_{index}.mat')[
                    'perfectChannel']
                dict_perfect_channel[item] = np.array(t_perfect_channel)

                t_noisy_channel = loadmat(f'{folder_name}dataset/{item}/noisyChannel{item}{db_value}dB_{index}.mat')[
                    'noisyChannel']
                dict_noisy_channel[item] = np.array(t_noisy_channel)

            noisy_channel = np.vstack([dict_noisy_channel[item] for item in channel_list])
            perfect_channel = np.vstack([dict_perfect_channel[item] for item in channel_list])

            all_noisy_channels.append(noisy_channel)
            all_perfect_channels.append(perfect_channel)

        final_noisy_channel = np.vstack(all_noisy_channels)
        final_perfect_channel = np.vstack(all_perfect_channels)

    scidx4x = np.hstack((range(-500, -2, 1), range(3, 501, 1)))
    scidx2x = np.hstack((range(-500, -2, 2), range(4, 501, 2)))
    target_scidx4x = range(-500, 501, 1)

    p_noisy_channel, p_perfect_channel = None, None
    if '4x80' == interpolate_mode:
        p_perfect_channel = yyy(scidx4x, target_scidx4x, final_perfect_channel)  # interpolate DC sub carriers
        p_noisy_channel = yyy(scidx4x, target_scidx4x, final_noisy_channel)  # interpolate DC sub carriers

    if '2x80' == interpolate_mode:
        p_perfect_channel = yyy(scidx4x, target_scidx4x, final_perfect_channel)  # interpolate DC sub carriers
        p_noisy_channel = xxx(scidx4x, scidx2x, target_scidx4x, final_noisy_channel)  # interpolate DC sub carriers

    return p_noisy_channel.reshape(data_len, 1, 1001), p_perfect_channel.reshape(data_len, 1, 1001)


'''
input: indices, prediction, x, y
output: SINR
'''
def get_evm_func_for_mid(indices, prediction, x, y):
    loss = x - y
    mse0 = np.zeros(int(x.shape[0] / 2))
    mse = np.zeros(int(x.shape[0] / 2))
    p = np.zeros(int(x.shape[0] / 2))
    evm0 = np.zeros(int(x.shape[0] / 2))
    evm = np.zeros(int(x.shape[0] / 2))
    for ii in range(0, x.shape[0], 2):
        # indices = scidx4x + 500
        se0r = (loss[ii, indices, :]) ** 2
        se0i = (loss[ii + 1, indices, :]) ** 2
        mse0[int(ii / 2)] = np.sum(se0r) + np.sum(se0i)
        ser = (prediction[ii, indices, :] - loss[ii, indices, :]) ** 2
        sei = (prediction[ii + 1, indices, :] - loss[ii + 1, indices, :]) ** 2
        mse[int(ii / 2)] = np.sum(ser) + np.sum(sei)
        p[int(ii / 2)] = np.sum((y[ii, indices, :]) ** 2) + np.sum((y[ii + 1, indices, :]) ** 2)
        evm0[int(ii / 2)] = mse0[int(ii / 2)] / p[int(ii / 2)]
        evm[int(ii / 2)] = mse[int(ii / 2)] / p[int(ii / 2)]


    eevm0 = - np.mean(10 * np.log10(evm0))
    eevm = - np.mean(10 * np.log10(evm))
    print('Original SNR: %.10f dB' % eevm0)
    print('Post-SNR:     %.10f dB' % eevm)

    return eevm - eevm0


def setup_logger(name, log_file, level=logging.INFO):
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger



'''
TRAINING FUNCTION
INPUT: m - index of RUNNING MODE, margin - number of EDGE BAND, db_value - value of SNR, channel_name - name of STANDARD CHANNEL
'''
def train_mlp(m, margin, db_value, channel_name):
    if margin == 0:
        print('Error! margin is 0')
        return

    mode_list = ['train', 'prediction']
    mode = mode_list[m]
    seed = 777777

    train_file_num = 8
    valid_file_num = 1
    test_file_num = 1
    file_num = train_file_num + valid_file_num + test_file_num  # data size
    batch_size = 64
    epoch = 2000
    lr = 0.001

    setting = '4x80'
    folder_name = '../'

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Baseline_MLP()
    model_name = model.name

    log_file = f"./logs/{model_name}_{db_value}_training.log"
    logger = setup_logger(model_name, log_file)

    logger.info(f"used device is: {device}")
    model = model.to(device)
    checkpoint_save_path = r"./checkpoint/{}/num{}epoch{}_{}_{}_{}dB_{}_margin{}.weights.pth".format(
        channel_name,
        file_num, epoch, model.name,
        setting, db_value,
        channel_name, margin)

    if mode == 'train':
        optimizer = Adam(model.parameters(), lr=lr)
        best_valid_loss = float('inf')

        x1, y1 = load_channel_data(folder_name, db_value, 1, train_file_num, [channel_name], setting)
        next_x_train = torch.from_numpy(x1).float()
        next_y_train = torch.from_numpy(y1).float()

        if margin < 0:
            next_train_dataset = CustomDataset(next_x_train[:, :, margin * 2:], next_y_train[:, :, margin * 2:])
        else:
            next_train_dataset = CustomDataset(next_x_train[:, :, :margin * 2], next_y_train[:, :, :margin * 2])

        train_loader = DataLoader(next_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        if valid_file_num == 0:
            x2, y2 = load_channel_data(folder_name, db_value, 1, 1, [channel_name], setting)
        else:
            x2, y2 = load_channel_data(folder_name, db_value, 1 + train_file_num, valid_file_num, [channel_name],
                                       setting)

        next_x_valid = torch.from_numpy(x2).float()
        next_y_valid = torch.from_numpy(y2).float()

        if margin < 0:
            next_valid_dataset = CustomDataset(next_x_valid[:, :, margin * 2:], next_y_valid[:, :, margin * 2:])
        else:
            next_valid_dataset = CustomDataset(next_x_valid[:, :, :margin * 2], next_y_valid[:, :, :margin * 2])
        valid_loader = DataLoader(next_valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        ######################TRAINING###########################
        for loop in range(epoch):
            model.train()
            train_loss_item = 0

            start_time = time.time()  # start time
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                if margin < 0:
                    labels = labels[:, :, margin:]
                else:
                    labels = labels[:, :, :margin]

                loss = my_loss(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss_item += loss.item()

            epoch_loss = train_loss_item / len(train_loader)
            end_time = time.time()  # end time
            logger.info(
                f"EPOCH: {loop}/{epoch}-Train Loss: {epoch_loss:.6f}   used time/epoch： {end_time - start_time:.2f}s")

            start_time = time.time()  # start time

            model.eval()
            valid_loss = 0
            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)
                    if margin < 0:
                        labels = labels[:, :, margin:]
                    else:
                        labels = labels[:, :, :margin]

                    loss = my_loss(outputs, labels)
                    valid_loss += loss.item()

            valid_loss /= (len(valid_loader))
            end_time = time.time()  # end time
            logger.info(
                f"EPOCH: {loop}/{epoch}-Valid Loss: {valid_loss:.6f}   used time/epoch： {end_time - start_time:.2f}s ")

            if checkpoint_save_path:
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(), checkpoint_save_path)
                    logger.info(f"Best model saved with validation loss: {best_valid_loss:.6f}")




if __name__ == "__main__":
    train_mlp(0, 10, 40, 'ChB')
    train_mlp(0, -10, 40, 'ChB')
    train_mlp(0, 10, 30, 'ChB')
    train_mlp(0, -10, 30, 'ChB')
    train_mlp(0,10,20, 'ChB')
    train_mlp(0, -10,20, 'ChB')
    train_mlp(0,10,10, 'ChB')
    train_mlp(0, -10,10, 'ChB')
    train_mlp(0,10,0, 'ChB')
    train_mlp(0, -10,0, 'ChB')

    train_mlp(0, 10, 40, 'ChD')
    train_mlp(0, -10, 40, 'ChD')
    train_mlp(0, 10, 30, 'ChD')
    train_mlp(0, -10, 30, 'ChD')
    train_mlp(0, 10, 20, 'ChD')
    train_mlp(0, -10, 20, 'ChD')
    train_mlp(0, 10, 10, 'ChD')
    train_mlp(0, -10, 10, 'ChD')
    train_mlp(0, 10, 0, 'ChD')
    train_mlp(0, -10, 0, 'ChD')