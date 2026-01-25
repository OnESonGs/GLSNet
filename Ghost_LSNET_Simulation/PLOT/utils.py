import numpy as np
from scipy.signal import wiener
import scipy.stats as stats

import torch.nn as nn
import gc

def calculate_mse(original, filtered):
    return np.mean((original - filtered) ** 2)

def WIENER_FILTER_LOAD(dbvalue, channel_name, XX, YY, loop):

    mse_list = []
    for idx in range(loop):
        indices = np.random.choice(8000, size=1000, replace=True)
        x = XX[indices]
        y = YY[indices]
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

        mse_list.append(mmse_opt)

        del x, y
        collected = gc.collect()
        print(f"COLLECTION NUMBER: {collected}")
    return mse_list



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
    input = torch.randn(8, 1, 1001)
    flops, params = profile(model, inputs=(input,))

    print(f"FLOPs: {flops / 1e9:.6f} GFLOPs, Params: {params / 1e6:.3f} M")

def mean_confidence_interval(data, confidence=0.99):

    mean = np.mean(data)
    sem = stats.sem(data)
    interval = sem * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
    return mean, mean - interval, mean + interval


if __name__ == "__main__":
    from LSNET.LSNet import LSNet_1D
    from GHOST_LSNET.Ghost_LSNet import LSGhostNet_1D
    from DNCNN.DNCNN import DNCNN
    get_flops(LSGhostNet_1D)
    get_flops(LSNet_1D)
    get_flops(DNCNN)


