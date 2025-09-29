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
class SkaFunction(Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        PyTorch 实现的 1D 空间自适应卷积前向传播
        """
        # 获取卷积核大小
        kernel_size = w.shape[2]
        pad = (kernel_size - 1) // 2
        # 保存反向传播所需的信息
        ctx.save_for_backward(x, w)
        ctx.pad = pad
        # 输入形状信息
        batch_size, in_channels, length = x.shape
        weight_channels = w.shape[1]
        # 对输入进行填充
        x_padded = F.pad(x, (pad, pad), mode='constant', value=0)
        # 展开输入张量
        x_unfold = x_padded.unfold(2, kernel_size, 1)  # [B, C, L, K]
        # 扩展权重张量以匹配输入通道数
        group_indices = torch.arange(in_channels, device=x.device) % weight_channels
        w_expanded = w[:, group_indices]  # [B, C, K, L]
        # 调整维度顺序以进行点积
        x_unfold = x_unfold.permute(0, 1, 3, 2)  # [B, C, K, L]
        # 计算输出: 点积并求和
        output = (x_unfold * w_expanded).sum(dim=2)  # [B, C, L]
        return output

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        """
        PyTorch 实现的 1D 空间自适应卷积反向传播
        """
        x, w = ctx.saved_tensors
        pad = ctx.pad
        # 输入形状信息
        batch_size, in_channels, length = x.shape
        weight_channels = w.shape[1]
        kernel_size = w.shape[2]
        # 对输入进行填充
        x_padded = F.pad(x, (pad, pad), mode='constant', value=0)
        # 展开输入张量
        x_unfold = x_padded.unfold(2, kernel_size, 1)  # [B, C, L, K]
        x_unfold = x_unfold.permute(0, 1, 3, 2)  # [B, C, K, L]
        # 扩展权重张量以匹配输入通道数
        group_indices = torch.arange(in_channels, device=x.device) % weight_channels
        w_expanded = w[:, group_indices]  # [B, C, K, L]

        # 计算输入梯度
        grad_input = None
        if ctx.needs_input_grad[0]:
            # 扩展梯度输出以匹配核大小
            grad_output_expanded = grad_output.unsqueeze(2)  # [B, C, 1, L]
            # 计算输入梯度
            grad_input_unfold = grad_output_expanded * w_expanded  # [B, C, K, L]
            # 调整维度顺序
            grad_input_unfold = grad_input_unfold.permute(0, 1, 3, 2)  # [B, C, L, K]
            # 折叠回原始形状（视为2D张量）
            grad_input_padded = F.fold(
                grad_input_unfold.contiguous().view(batch_size, in_channels * kernel_size, length),
                output_size=(1, length + 2 * pad),  # 二维尺寸
                kernel_size=(1, kernel_size),
                padding=(0, 0),
                stride=(1, 1)
            )
            # 压缩高度维度 [B, C, 1, L+2*pad] -> [B, C, L+2*pad]
            grad_input_padded = grad_input_padded.squeeze(2)
            # 移除填充
            grad_input = grad_input_padded[:, :, pad:pad + length]
        # 计算权重梯度
        grad_weight = None
        if ctx.needs_input_grad[1]:
            # 扩展梯度输出
            grad_output_expanded = grad_output.unsqueeze(2)  # [B, C, 1, L]
            # 计算每个通道的权重梯度
            grad_weight_per_channel = x_unfold * grad_output_expanded  # [B, C, K, L]
            # 初始化权重梯度张量
            grad_weight = torch.zeros_like(w)
            # 对每个权重通道求和
            for group_idx in range(weight_channels):
                # 获取属于当前组的通道掩码
                mask = (group_indices == group_idx)
                # 对组内所有通道求和
                if mask.any():
                    grad_weight[:, group_idx] = grad_weight_per_channel[:, mask].sum(dim=1)

        return grad_input, grad_weight, None, None

class SKA1d(nn.Module):
    def __init__(self, in_channels, groups=None):
        """
        1D 空间自适应卷积模块
        参数:
            in_channels: 输入通道数
            kernel_size: 卷积核大小
            groups: 分组数 (权重通道数)
        """
        super().__init__()
        self.groups = groups or max(1, in_channels // 4)  # 默认分组数
        # 权重张量形状: [1, groups, kernel_size, 1]
        # 在 forward 中扩展为 [batch_size, groups, kernel_size, length]

    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        参数:
            x: 输入张量, 形状为 [batch_size, in_channels, length]
        返回:
            输出张量, 形状为 [batch_size, in_channels, length]
        """
        # 扩展权重以匹配批量大小和长度
        batch_size, _, length = x.shape

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
        N = len(points)
        attention_offsets = {}
        idxs = []

        # 生成1D相对位置偏移
        for p1 in points:
            for p2 in points:
                # 1D位置差（绝对值）
                offset = abs(p1 - p2)
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])

        # 可学习的位置偏置参数
        self.attention_biases = nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))

        # 注册缓冲区保存位置索引
        # self.register_buffer('attention_bias_idxs',
        #                      torch.LongTensor(idxs).view(N, N))

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

class RepVGGDW(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv1d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = Conv1d_BN(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed

    def forward(self, x):
        return self.conv(x) + self.conv1(x) + x

    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()
        conv1 = self.conv1.fuse()

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [1, 1, 1])

        identity = torch.nn.functional.pad(torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, device=conv1_w.device),
                                           [1, 1, 1])

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)
        return conv

# SE模块（Squeeze-and-Excitation）
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channel, channel // reduction, kernel_size=1, bias=False),
            nn.BatchNorm1d(channel // reduction),
            nn.ReLU(),
            nn.Conv1d(channel // reduction, channel, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_se = x.mean((2), keepdim=True)
        x_se = 0.5 * x_se + 0.5 * x.amax((2), keepdim=True)
        x_se = self.fc(x_se)
        return x * x_se

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

class FFN(torch.nn.Module):
    def __init__(self, ed, h):
        super().__init__()
        self.pw1 = Conv1d_BN(ed, h)
        self.act = torch.nn.ReLU()
        self.pw2 = Conv1d_BN(h, ed, bn_weight_init=0)

    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
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
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups,
            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

class LKP1d(nn.Module):
    def __init__(self, dim, lks, sks, groups):
        super().__init__()
        self.cv1 = Conv1d_BN(dim, dim // 2) # (B, dim/2, L)
        self.act = nn.SiLU()  # nn.ReLU()
        self.cv2 = Conv1d_BN(dim // 2, dim // 2, ks=lks, pad=(lks - 1) // 2, groups=dim // 2)
        self.cv3 = Conv1d_BN(dim // 2, dim // 2)
        self.cv4 = nn.Conv1d(dim // 2, sks * dim // groups, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=dim // groups, num_channels=sks * dim // groups)

        self.sks = sks
        self.groups = groups
        self.dim = dim

    def forward(self, x):
        x = self.act(self.cv3(self.cv2(self.act(self.cv1(x)))))
        w = self.norm(self.cv4(x))
        b, _, L = w.size()
        w = w.view(b, self.dim // self.groups, self.sks, L)   # (B, dim/2, sks, L)
        return w


class LSConv(nn.Module):
    def __init__(self, dim):
        super(LSConv, self).__init__()
        self.lkp =  LKP1d(dim, lks=7, sks=3, groups=8)
        self.ska = SKA1d(dim)
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x):
        return self.bn(self.ska(x, self.lkp(x))) + x


class LSNet_1D(nn.Module):
    def __init__(self, ch=64):
        super().__init__()
        self.name = 'LSNet-1D'

        # 输入处理
        self.stem = nn.Sequential(
            nn.Conv1d(1, ch, kernel_size=5, padding=(5-1)//2),
            nn.ReLU(),
        )

        # 特征提取路径（通道数逐渐增加）
        self.Block1 = nn.Sequential(
            RepVGGDW(ch),
            SEBlock(ch),
        )
        self.ffn1 = Residual(FFN(ch, int(ch * 2)))

        # 特征融合
        self.Block2 = nn.Sequential(
            nn.Conv1d(ch, 2 * ch, 3, padding=(3-1)//2),
            nn.Conv1d(2 * ch, 2 * ch, 1),
        )
        # 特征提取路径（通道数逐渐增加）
        self.Block3 = nn.Sequential(
            RepVGGDW(2 * ch),
            SEBlock(2 * ch),
        )
        self.ffn2 = Residual(FFN(2 * ch, int(ch * 4)))
        self.LSConv = LSConv(ch*2)

        self.ffn3 = Residual(Attention1d(dim=ch * 2, key_dim=ch // 2, num_heads=8,
                 attn_ratio=4,
                 resolution=1001))

        # 输出层
        self.head = nn.Sequential(
            nn.Conv1d(2 * ch, ch, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(ch, 1, 1)  # 最终输出1个通道，保持与输入相同的形状
        )

    def forward(self, x):
        # 输入形状: (64, 1, 1001)
        batch_size, _, seq_length = x.shape

        # 特征提取
        x = self.stem(x)  # (64, 64, 1001)

        # 第一阶段处理
        x = self.Block1(x)  # (64, 64, 1001)
        x = self.ffn1(x)  # (64, 64, 1001)

        # 第二阶段处理 - 增加通道数
        x = self.Block2(x)  # (64, 128, 1001)

        # 第三阶段处理
        x = self.Block3(x)  # (64, 128, 1001)
        x = self.ffn2(x)  # (64, 128, 1001)

        # 应用LS卷积
        x = self.LSConv(x)  # (64, 128, 1001)

        # 应用注意力机制
        x = self.ffn3(x)  # (64, 128, 1001)

        # 输出层 - 回归预测
        output = self.head(x)  # (64, 1, 1001)

        return output

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

if __name__ == "__main__":
    ######################################参数配置#####################################
    mode_list = ['train', 'prediction']
    m = int(input("请输入训练模式:"))
    mode = mode_list[m]  # 训练模式

    channel_num = 64  # 通道数

    seed = 777777  # 随机数种子

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
    bk_ver = 'LSNET'

    folder_name = 'D:/AI_Filter/'
    ######################################网络结构#####################################
    np.random.seed(seed)  # 设置numpy
    torch.manual_seed(seed)  # 设置torch
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检查是否有可用的GPU

    # 创建模型实例以获取模型名称
    model = LSNet_1D(ch=channel_num)
    model_name = model.name

    # 设置日志记录器，使用模型名作为日志文件名
    log_file = f"./logs/{model_name}_{db_value}_training.log"
    logger = setup_logger(model_name, log_file)

    logger.info(f"used device is: {device}")
    model = model.to(device)  # 模型选择(to gpu)
    checkpoint_save_path = r"./checkpoint/{}/{}_margin{}_num{}_epoch{}_model{}_setting{}_dB{}_channel{}.weights.pth".format(channel_name,
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
