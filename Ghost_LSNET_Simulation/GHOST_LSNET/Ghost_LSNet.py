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
class SkaFunction(Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        kernel_size = w.shape[2]
        pad = (kernel_size - 1) // 2
        ctx.save_for_backward(x, w)
        ctx.pad = pad
        batch_size, in_channels, length = x.shape
        weight_channels = w.shape[1]
        x_padded = F.pad(x, (pad, pad), mode='constant', value=0)
        x_unfold = x_padded.unfold(2, kernel_size, 1)
        group_indices = torch.arange(in_channels, device=x.device) % weight_channels
        w_expanded = w[:, group_indices]
        x_unfold = x_unfold.permute(0, 1, 3, 2)
        output = (x_unfold * w_expanded).sum(dim=2)
        return output

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        x, w = ctx.saved_tensors
        pad = ctx.pad
        batch_size, in_channels, length = x.shape
        weight_channels = w.shape[1]
        kernel_size = w.shape[2]
        x_padded = F.pad(x, (pad, pad), mode='constant', value=0)
        x_unfold = x_padded.unfold(2, kernel_size, 1)
        x_unfold = x_unfold.permute(0, 1, 3, 2)
        group_indices = torch.arange(in_channels, device=x.device) % weight_channels
        w_expanded = w[:, group_indices]

        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_output_expanded = grad_output.unsqueeze(2)
            grad_input_unfold = grad_output_expanded * w_expanded
            grad_input_unfold = grad_input_unfold.permute(0, 1, 3, 2)
            grad_input_padded = F.fold(
                grad_input_unfold.contiguous().view(batch_size, in_channels * kernel_size, length),
                output_size=(1, length + 2 * pad),
                kernel_size=(1, kernel_size),
                padding=(0, 0),
                stride=(1, 1)
            )
            grad_input_padded = grad_input_padded.squeeze(2)
            grad_input = grad_input_padded[:, :, pad:pad + length]
        grad_weight = None
        if ctx.needs_input_grad[1]:
            grad_output_expanded = grad_output.unsqueeze(2)
            grad_weight_per_channel = x_unfold * grad_output_expanded
            grad_weight = torch.zeros_like(w)
            for group_idx in range(weight_channels):
                mask = (group_indices == group_idx)
                if mask.any():
                    grad_weight[:, group_idx] = grad_weight_per_channel[:, mask].sum(dim=1)

        return grad_input, grad_weight, None, None

class SKA1d(nn.Module):
    def __init__(self, in_channels, groups=None):
        super().__init__()
        self.groups = groups or max(1, in_channels // 4)

    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return SkaFunction.apply(x, weight)

class Attention1d(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 resolution=16):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2
        self.qkv = Conv1d_BN(dim, h, ks=1)
        self.proj = torch.nn.Sequential(torch.nn.ReLU(), Conv1d_BN(
            self.dh, dim, bn_weight_init=0))
        self.dw = Conv1d_BN(nh_kd, nh_kd, 3, 1, 1, groups=nh_kd)
        points = range(resolution)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = abs(p1 - p2)
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                         torch.LongTensor([attention_offsets[abs(i-j)] for i in points for j in points])
                         .view(resolution, resolution)
                         )

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):
        B, _, L = x.shape
        N = L
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, -1, L).split([self.nh_kd, self.nh_kd, self.dh], dim=1)
        q = self.dw(q)
        q, k, v = q.view(B, self.num_heads, -1, N), k.view(B, self.num_heads, -1, N), v.view(B, self.num_heads, -1, N)
        attn = (
                (q.transpose(-2, -1) @ k) * self.scale
                +
                (self.attention_biases[:, self.attention_bias_idxs]
                 if self.training else self.ab)
        )
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).reshape(B, -1, L)
        x = self.proj(x)
        return x

class Conv1d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv1d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm1d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv1d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride(0), padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups,
            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

class LKP1d(nn.Module):
    def __init__(self, dim, lks, sks, groups):
        super().__init__()
        self.cv1 = Conv1d_BN(dim, dim // 2) # (B, dim/2, L)
        self.act = nn.ReLU()
        self.cv2 = Conv1d_BN(dim // 2, dim // 2, ks=lks, pad=(lks - 1) // 2, groups=dim // 2)
        self.cv3 = Conv1d_BN(dim // 2, dim // 2)

        self.residual = nn.Conv1d(dim // 2, dim // 2, kernel_size=1)

        self.cv4 = nn.Conv1d(dim // 2, sks * dim // groups, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=dim // groups, num_channels=sks * dim // groups)

        self.sks = sks
        self.groups = groups
        self.dim = dim

    def forward(self, x):
        residual = self.cv1(x)
        x = self.act(self.cv3(self.cv2(self.act(residual))))
        x = x + self.residual(residual)

        w = self.norm(self.cv4(x))
        b, _, L = w.size()
        w = w.view(b, self.dim // self.groups, self.sks, L)   # (B, dim/2, sks, L)
        return w



class LSConv(nn.Module):
    def __init__(self, dim, ks):
        super(LSConv, self).__init__()
        self.ska_cheap = SKA1d(dim)
        self.lkp_cheap =  LKP1d(dim, lks=ks, sks=2*(ks//4)+1, groups=8)

        self.ska_primary = SKA1d(dim)
        self.lkp_primary = LKP1d(dim, lks=7, sks=3, groups=8)
        self.bn = nn.BatchNorm1d(dim*2)
        self.act = nn.SiLU()

    def forward(self, x):
        x1 = self.ska_primary(x, self.lkp_primary(x))
        x2 = self.ska_cheap(x1, self.lkp_cheap(x1))

        out = torch.cat([x1, x2], dim=1)
        return self.act(self.bn(out))


class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)

class LSGhostNet_1D(nn.Module):
    def __init__(self, input_length=1001, ch=64):
        super().__init__()
        self.name = 'LSGhostNet-1D'

        self.stem = nn.Sequential(
            nn.Conv1d(1, ch, kernel_size=9, padding=(9-1)//2),
            nn.BatchNorm1d(ch),
            nn.ReLU(inplace=True)
        )

        self.feaupsample1 = LSConv(ch, 5)  # ch -> 2*ch

        self.ffn3 = Residual(Attention1d(dim=ch * 2, key_dim=ch // 2, num_heads=8,
                                         attn_ratio=2,
                                         resolution=1001))

        self.head = nn.Sequential(
            nn.Conv1d(2 * ch, ch, 3, padding=1),
            nn.BatchNorm1d(ch),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(ch, 1, 1)
        )

    def forward(self, x):
        x1 = self.stem(x)  # [B,64,L]
        x2 = self.feaupsample1(x1)  # [B,128,L]

        y = self.ffn3(x2)
        output = self.head(y)  # [B,1,L]
        return output

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
def train_glsnet(m, margin, db_value, channel_name):
    mode_list = ['train', 'prediction']
    mode = mode_list[m]
    channel_num = 64
    seed = 777777

    train_file_num = 8
    valid_file_num = 1
    test_file_num = 1
    file_num = train_file_num + valid_file_num + test_file_num  # data size
    batch_size = 64
    epoch = 200
    lr = 0.001

    setting = '4x80'
    bk_ver = 'GHOST_LSNET'
    folder_name = '../'

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSGhostNet_1D(ch=channel_num)
    model_name = model.name

    log_file = f"./logs/{model_name}_{db_value}_training.log"
    logger = setup_logger(model_name, log_file)

    logger.info(f"used device is: {device}")
    model = model.to(device)
    checkpoint_save_path = r"./checkpoint/{}/{}_margin{}_num{}_epoch{}_model{}_setting{}_dB{}_channel{}.weights.pth".format(
        channel_name,
        bk_ver, margin,
        file_num, epoch, model.name,
        setting, db_value,
        channel_name)

    if mode == 'train':
        optimizer = Adam(model.parameters(), lr=lr)
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

            valid_loss /= (len(valid_loader))
            end_time = time.time()  # end time
            logger.info(
                f"EPOCH: {loop}/{epoch}-Valid Loss: {valid_loss:.6f}   used time/epoch： {end_time - start_time:.2f}s ")

            if checkpoint_save_path:
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(), checkpoint_save_path)
                    logger.info(f"Best model saved with validation loss: {best_valid_loss:.6f}")

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
    model.load_state_dict(torch.load(checkpoint_save_path, weights_only=True, map_location=device))

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

    scidx4x = np.hstack((range(-500, -2, 1), range(3, 501, 1)))
    indices = scidx4x + 500

    logger.info("MID BAND:")
    if margin < 0:
        get_evm_func_for_mid(indices[-1 * margin:margin], p_test, x_test, y_test)
    elif margin > 0:
        get_evm_func_for_mid(indices[margin:-1 * margin], p_test, x_test, y_test)
    else:
        get_evm_func_for_mid(indices[:], p_test, x_test, y_test)


if __name__ == "__main__":
    train_glsnet(0, 10, 40, 'ChB')
    train_glsnet(0, 10, 30, 'ChB')
    train_glsnet(0, 10, 20, 'ChB')
    train_glsnet(0, 10, 10, 'ChB')
    train_glsnet(0, 10, 0, 'ChB')

    train_glsnet(0, 10, 40, 'ChD')
    train_glsnet(0, 10, 30, 'ChD')
    train_glsnet(0, 10, 20, 'ChD')
    train_glsnet(0, 10, 10, 'ChD')
    train_glsnet(0, 10, 0, 'ChD')