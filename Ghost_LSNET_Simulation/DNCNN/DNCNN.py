import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.data import DataLoader

import torch
import numpy as np
import time
import logging
from pathlib import Path

#######################神经网络###############################

class DNCNN(nn.Module):
    def __init__(self):
        super(DNCNN, self).__init__()
        self.name = 'DNCNN'
        # 卷积层定义
        self.c1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)  # 等效于padding='same'
        self.a1 = nn.ReLU(inplace=True)
        self.c2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.b2 = nn.BatchNorm1d(num_features=64)
        self.a2 = nn.ReLU(inplace=True)
        self.c3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.b3 = nn.BatchNorm1d(num_features=64)
        self.a3 = nn.ReLU(inplace=True)
        self.c4 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.b4 = nn.BatchNorm1d(num_features=64)
        self.a4 = nn.ReLU(inplace=True)
        self.c5 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.b5 = nn.BatchNorm1d(num_features=64)
        self.a5 = nn.ReLU(inplace=True)
        self.c6 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.b6 = nn.BatchNorm1d(num_features=64)
        self.a6 = nn.ReLU(inplace=True)
        self.c7 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.b7 = nn.BatchNorm1d(num_features=64)
        self.a7 = nn.ReLU(inplace=True)
        self.c8 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.b8 = nn.BatchNorm1d(num_features=64)
        self.a8 = nn.ReLU(inplace=True)
        self.c9 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.b9 = nn.BatchNorm1d(num_features=64)
        self.a9 = nn.ReLU(inplace=True)
        self.c10 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.b10 = nn.BatchNorm1d(num_features=64)
        self.a10 = nn.ReLU(inplace=True)
        self.c11 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.b11 = nn.BatchNorm1d(num_features=64)
        self.a11 = nn.ReLU(inplace=True)
        self.c12 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.b12 = nn.BatchNorm1d(num_features=64)
        self.a12 = nn.ReLU(inplace=True)
        self.c13 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.b13 = nn.BatchNorm1d(num_features=64)
        self.a13 = nn.ReLU(inplace=True)
        self.c14 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.b14 = nn.BatchNorm1d(num_features=64)
        self.a14 = nn.ReLU(inplace=True)
        self.c15 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.b15 = nn.BatchNorm1d(num_features=64)
        self.a15 = nn.ReLU(inplace=True)
        self.c16 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.b16 = nn.BatchNorm1d(num_features=64)
        self.a16 = nn.ReLU(inplace=True)
        self.c17 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.b17 = nn.BatchNorm1d(num_features=64)
        self.a17 = nn.ReLU(inplace=True)
        self.c18 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.b18 = nn.BatchNorm1d(num_features=64)
        self.a18 = nn.ReLU(inplace=True)
        self.c19 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.b19 = nn.BatchNorm1d(num_features=64)
        self.a19 = nn.ReLU(inplace=True)
        self.c20 = nn.Conv1d(in_channels=64, out_channels=1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.a1(self.c1(x))
        x = self.a2(self.b2(self.c2(x)))
        x = self.a3(self.b3(self.c3(x)))
        x = self.a4(self.b4(self.c4(x)))
        x = self.a5(self.b5(self.c5(x)))
        x = self.a6(self.b6(self.c6(x)))
        x = self.a7(self.b7(self.c7(x)))
        x = self.a8(self.b8(self.c8(x)))
        x = self.a9(self.b9(self.c9(x)))
        x = self.a10(self.b10(self.c10(x)))
        x = self.a11(self.b11(self.c11(x)))
        x = self.a12(self.b12(self.c12(x)))
        x = self.a13(self.b13(self.c13(x)))
        x = self.a14(self.b14(self.c14(x)))
        x = self.a15(self.b15(self.c15(x)))
        x = self.a16(self.b16(self.c16(x)))
        x = self.a17(self.b17(self.c17(x)))
        x = self.a18(self.b18(self.c18(x)))
        x = self.a19(self.b19(self.c19(x)))
        y = self.c20(x)
        return y


# 定义损失函数和优化器
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

        # 合并所有读取的文件数据
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

        # 合并所有读取的文件数据
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

    eevm0 = 10 * np.log10(np.mean(evm0))
    eevm = 10 * np.log10(np.mean(evm))
    print('(线性)原EVM: %.10f' % eevm0)
    print('(线性)后EVM: %.10f' % eevm)

    eevm0 = np.mean(10 * np.log10(evm0))
    eevm = np.mean(10 * np.log10(evm))
    print('(非线性)原EVM: %.10f' % eevm0)
    print('(非线性)后EVM: %.10f' % eevm)



# 配置日志
def setup_logger(name, log_file, level=logging.INFO):
    """设置日志记录器，同时输出到控制台和文件"""
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)

    # 控制台处理器
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重复添加处理器
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger


def train_dncnn(m, margin, db_value, channel_name):
    ######################################参数配置#####################################
    mode_list = ['train', 'prediction']
    # m = int(input("请输入训练模式:"))
    mode = mode_list[m]  # 训练模式

    channel_num = 64  # 通道数

    seed = 777777  # 随机数种子

    # margin = int(input("请输入边带数:"))  # edge band
    # db_value = input("请输入SNR:")  # snr

    train_file_num = 8
    valid_file_num = 1
    test_file_num = 1
    file_num = train_file_num + valid_file_num + test_file_num  # data size

    batch_size = 64
    epoch = 200
    lr = 0.001

    setting = '4x80'
    # channel_name = input('请输入信道模型: ')
    bk_ver = 'DNCNN'

    folder_name = 'D:/AI_Filter/'
    ######################################网络结构#####################################
    np.random.seed(seed)  # 设置numpy
    torch.manual_seed(seed)  # 设置torch
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检查是否有可用的GPU

    # 创建模型实例以获取模型名称
    model = DNCNN()
    model_name = model.name

    # 设置日志记录器，使用模型名作为日志文件名
    log_file = f"./logs/{model_name}_{db_value}_training.log"
    logger = setup_logger(model_name, log_file)

    logger.info(f"used device is: {device}")
    model = model.to(device)  # 模型选择(to gpu)
    checkpoint_save_path = r"./checkpoint/{}_margin{}_num{}_epoch{}_model{}_setting{}_dB{}_channel{}.weights.pth".format(
        bk_ver, margin,
        file_num, epoch, model.name,
        setting, db_value,
        channel_name)

    if mode == 'train':
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        best_valid_loss = float('inf')

        x1, y1 = load_channel_data(folder_name, db_value, 1, train_file_num, [channel_name], setting)
        next_x_train = torch.from_numpy(x1).float()
        next_y_train = torch.from_numpy(y1).float()
        next_train_dataset = CustomDataset(next_x_train, next_y_train)
        train_loader = DataLoader(next_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        if valid_file_num == 0:
            x2, y2 = load_channel_data(folder_name, db_value, 1, 1, [channel_name], setting)
        else:
            x2, y2 = load_channel_data(folder_name, db_value, 1 + train_file_num, valid_file_num, [channel_name],
                                       setting)
        next_x_valid = torch.from_numpy(x2).float()
        next_y_valid = torch.from_numpy(y2).float()
        next_valid_dataset = CustomDataset(next_x_valid, next_y_valid)
        valid_loader = DataLoader(next_valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        ######################TRAINING###########################
        for loop in range(epoch):
            # 训练阶段
            model.train()
            train_loss_item = 0

            start_time = time.time()  # start time
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                if margin == 0:
                    pass
                elif margin > 0:
                    labels = labels[:, :, margin:-margin]
                    outputs = outputs[:, :, margin:-margin]
                else:
                    labels = labels[:, :, -margin:margin]
                    outputs = outputs[:, :, -margin:margin]

                loss = my_loss(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss_item += loss.item()

            # 计算平均 loss
            epoch_loss = train_loss_item / len(train_loader)
            end_time = time.time()  # end time
            logger.info(
                f"EPOCH: {loop}/{epoch}-Train Loss: {epoch_loss:.4f}   used time/epoch： {end_time - start_time:.2f}s")

            start_time = time.time()  # start time
            # 验证阶段
            model.eval()
            valid_loss = 0
            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)
                    if margin == 0:
                        pass
                    elif margin > 0:
                        labels = labels[:, :, margin:-margin]
                        outputs = outputs[:, :, margin:-margin]
                    else:
                        labels = labels[:, :, -margin:margin]
                        outputs = outputs[:, :, -margin:margin]

                    loss = my_loss(outputs, labels)
                    valid_loss += loss.item()

            # 计算平均 loss
            valid_loss /= (len(valid_loader))
            end_time = time.time()  # end time
            logger.info(
                f"EPOCH: {loop}/{epoch}-Valid Loss: {valid_loss:.4f}   used time/epoch： {end_time - start_time:.2f}s ")
            # 保存最佳模型
            if checkpoint_save_path:
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(), checkpoint_save_path)
                    logger.info(f"Best model saved with validation loss: {best_valid_loss:.4f}")

        ######################PREDICTION###########################

    if valid_file_num == 0:
        x, y = load_channel_data(folder_name, db_value, 1, 1, [channel_name], setting)
    else:
        x, y = load_channel_data(folder_name, db_value, 1 + train_file_num + valid_file_num, test_file_num,
                                 [channel_name], setting)

    next_x_test = torch.from_numpy(x).float().to(device)
    next_y_test = torch.from_numpy(y).float().to(device)
    next_test_dataset = CustomDataset(next_x_test, next_y_test)
    test_loader = DataLoader(next_test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_p = []

    logger.info(f"Loading best model from {checkpoint_save_path}")
    model.load_state_dict(torch.load(checkpoint_save_path, weights_only=True))
    # 测试阶段
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            batch_p = model(inputs)
            all_p.append(batch_p.cpu().detach())

    p = torch.cat(all_p, dim=0)
    data_len = test_file_num * 8000

    x_test = x.reshape(data_len, 1001, 1)
    y_test = y.reshape(data_len, 1001, 1)
    p_test = p.numpy().reshape(data_len, 1001, 1)
    loss = x_test - y_test

    scidx4x = np.hstack((range(-500, -2, 1), range(3, 501, 1)))
    indices = scidx4x + 500

    logger.info("中间带:")
    if margin < 0:
        get_evm_func_for_mid(indices[-1 * margin:margin], p_test, x_test, y_test)
    elif margin > 0:
        get_evm_func_for_mid(indices[margin:-1 * margin], p_test, x_test, y_test)
    else:
        get_evm_func_for_mid(indices[:], p_test, x_test, y_test)

if __name__ == "__main__":
    # train_dncnn(0,0,20, 'ChB')
    # train_dncnn(0,0,10, 'ChB')
    # train_dncnn(0,0,0, 'ChB')

    train_dncnn(0, 0, 40, 'ChD')
    # train_dncnn(0, 0, 30, 'ChD')
    # train_dncnn(0, 0, 20, 'ChD')
    # train_dncnn(0, 0, 10, 'ChD')
    # train_dncnn(0, 0, 0, 'ChD')

if __name__ == "__main1111__":
    ######################################参数配置#####################################
    mode_list = ['train', 'prediction']
    m = int(input("请输入训练模式:"))
    mode = mode_list[m]  # 训练模式

    channel_num = 64  # 通道数

    seed = 666666  # 随机数种子 # 777777 ChD 情况下 不收敛

    margin = int(input("请输入边带数:"))  # edge band
    db_value = input("请输入SNR:")  # snr

    train_file_num = 8
    valid_file_num = 1
    test_file_num = 1
    file_num = train_file_num + valid_file_num + test_file_num  # data size

    batch_size = 64
    epoch = 200
    lr = 0.001

    setting = '4x80'
    channel_name = input('请输入信道模型: ')
    bk_ver = 'DNCNN'

    folder_name = 'D:/AI_Filter/'
    ######################################网络结构#####################################
    np.random.seed(seed)  # 设置numpy
    torch.manual_seed(seed)  # 设置torch
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检查是否有可用的GPU

    # 创建模型实例以获取模型名称
    model = DNCNN()
    model_name = model.name

    # 设置日志记录器，使用模型名作为日志文件名
    log_file = f"./logs/{model_name}_{db_value}_training.log"
    logger = setup_logger(model_name, log_file)

    logger.info(f"used device is: {device}")
    model = model.to(device)  # 模型选择(to gpu)
    checkpoint_save_path = r"./checkpoint/{}_margin{}_num{}_epoch{}_model{}_setting{}_dB{}_channel{}.weights.pth".format(
        bk_ver, margin,
        file_num, epoch, model.name,
        setting, db_value,
        channel_name)

    if mode == 'train':
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        best_valid_loss = float('inf')

        x1, y1 = load_channel_data(folder_name, db_value, 1, train_file_num, [channel_name], setting)
        next_x_train = torch.from_numpy(x1).float()
        next_y_train = torch.from_numpy(y1).float()
        next_train_dataset = CustomDataset(next_x_train, next_y_train)
        train_loader = DataLoader(next_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        if valid_file_num == 0:
            x2, y2 = load_channel_data(folder_name, db_value, 1, 1, [channel_name], setting)
        else:
            x2, y2 = load_channel_data(folder_name, db_value, 1 + train_file_num, valid_file_num, [channel_name], setting)
        next_x_valid = torch.from_numpy(x2).float()
        next_y_valid = torch.from_numpy(y2).float()
        next_valid_dataset = CustomDataset(next_x_valid, next_y_valid)
        valid_loader = DataLoader(next_valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        ######################TRAINING###########################
        for loop in range(epoch):
            # 训练阶段
            model.train()
            train_loss_item = 0

            start_time = time.time()  # start time
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                if margin == 0:
                    pass
                elif margin > 0:
                    labels = labels[:, :, margin:-margin]
                    outputs = outputs[:, :, margin:-margin]
                else:
                    labels = labels[:, :, -margin:margin]
                    outputs = outputs[:, :, -margin:margin]

                loss = my_loss(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss_item += loss.item()

            # 计算平均 loss
            epoch_loss = train_loss_item / len(train_loader)
            end_time = time.time()  # end time
            logger.info(
                f"EPOCH: {loop}/{epoch}-Train Loss: {epoch_loss:.4f}   used time/epoch： {end_time - start_time:.2f}s")

            start_time = time.time()  # start time
            # 验证阶段
            model.eval()
            valid_loss = 0
            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)
                    if margin == 0:
                        pass
                    elif margin > 0:
                        labels = labels[:, :, margin:-margin]
                        outputs = outputs[:, :, margin:-margin]
                    else:
                        labels = labels[:, :, -margin:margin]
                        outputs = outputs[:, :, -margin:margin]

                    loss = my_loss(outputs, labels)
                    valid_loss += loss.item()

            # 计算平均 loss
            valid_loss /= (len(valid_loader))
            end_time = time.time()  # end time
            logger.info(
                f"EPOCH: {loop}/{epoch}-Valid Loss: {valid_loss:.4f}   used time/epoch： {end_time - start_time:.2f}s ")
            # 保存最佳模型
            if checkpoint_save_path:
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(), checkpoint_save_path)
                    logger.info(f"Best model saved with validation loss: {best_valid_loss:.4f}")

        ######################PREDICTION###########################

    if valid_file_num == 0:
        x, y = load_channel_data(folder_name, db_value, 1, 1, [channel_name], setting)
    else:
        x, y = load_channel_data(folder_name, db_value, 1 + train_file_num + valid_file_num, test_file_num, [channel_name], setting)

    next_x_test = torch.from_numpy(x).float().to(device)
    next_y_test = torch.from_numpy(y).float().to(device)
    next_test_dataset = CustomDataset(next_x_test, next_y_test)
    test_loader = DataLoader(next_test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_p = []

    logger.info(f"Loading best model from {checkpoint_save_path}")
    model.load_state_dict(torch.load(checkpoint_save_path, weights_only=True))
    # 测试阶段
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            batch_p = model(inputs)
            all_p.append(batch_p.cpu().detach())

    p = torch.cat(all_p, dim=0)
    data_len = test_file_num * 8000

    x_test = x.reshape(data_len, 1001, 1)
    y_test = y.reshape(data_len, 1001, 1)
    p_test = p.numpy().reshape(data_len, 1001, 1)
    loss = x_test - y_test

    scidx4x = np.hstack((range(-500, -2, 1), range(3, 501, 1)))
    indices = scidx4x + 500

    logger.info("中间带:")
    if margin < 0:
        get_evm_func_for_mid(indices[-1 * margin:margin], p_test, x_test, y_test)
    elif margin > 0:
        get_evm_func_for_mid(indices[margin:-1 * margin], p_test, x_test, y_test)
    else:
        get_evm_func_for_mid(indices[:], p_test, x_test, y_test)
