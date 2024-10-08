import numpy as np
from data.MyDataLoader import get_dataloader
from feature_env import Evaluator, base_path
from utils.logger import info, error
from autos.train import autos_selection
from types import SimpleNamespace
import torch
from torch import nn
import logging
import math
import random
import os
import sys
import pickle
import warnings

# 数据集文件路径
data_path = "data/datasets.json"
origin_latency = 90.206 * 0.9
origin_energy = 0.00628754586336555 * 0.8  

batch_size = 32

# performace 计算函数:f = (1 / ppl) * (T / t) ** (0 if T < t else a) * (E / e) ** (0 if E < e else b)
def get_performance(ppl: float, T: float, t: float, 
                    E: float, e: float, a: int, b: int) -> float:
        return (1 / ppl) * (T / t) ** (0 if T < t else a) * (E / e) ** (0 if E < e else b)

def get_T(a: int, b: int, dataloader, T=50.0) -> float:
    pre_performance = 0
    delta = 0.5
    for data in dataloader:
        performance = 0
        _, ppl, t, e =  data
        for i in range(len(ppl)):
            performance += get_performance(ppl[i], T, t[i], e[i], e[i], a, b)
        if performance < pre_performance:
            T += delta
        pre_performance = performance
    return T

def get_E(a: int, b: int, dataloader, E=0.0) ->float:
    pre_performance = 0
    delta = 0.001
    for data in dataloader:
        performance = 0
        _, ppl, t, e =  data
        for i in range(len(ppl)):
            performance += get_performance(ppl[i], t[i], t[i], E, e[i], a, b)
        if performance < pre_performance:
            E += delta
        pre_performance = performance
    return E

def gen_device_selection(device_eval_):   
        dataloader = get_dataloader(data_path, batch_size=batch_size, shuffle=True)
        a, b = 1, 1
        # T, E = get_T(a, b, dataloader), get_E(a, b, dataloader)
        # print(f"T:{T}, E:{E}")
        T, E = origin_latency, origin_energy
        for data in dataloader:
            pruning_ratio, ppl, t, e =  data
            for i in range(len(ppl)):
                # if t[i] > origin_latency:
                #     continue
                performances = get_performance(ppl[i], T, t[i], E, e[i], a, b)
                # performances = get_performance(1, 1, 1, E, e[i], 1, b) # Min Energy(Mp)
                pruning_ratio_to_idx = torch.tensor([(pruning_ratio[i][j] * 1000 + j * 1000) for j in range(32)])
                device_eval_._store_history(pruning_ratio_to_idx, performances)
                
def choose_clients():
    # 采集信息
    device_eval = Evaluator()
    gen_device_selection(device_eval)
    #　保存pruning_ratio-performance对
    file_path = f"{base_path}/history/pruning_ratio-performance.pkl"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(f'{base_path}/history/pruning_ratio-performance.pkl', 'wb') as f: 
        pickle.dump(device_eval, f)
    print("=========================开始训练选择网络与最优剪枝生成=========================")
    # 训练选择网络与最优剪枝生成
    autos_selection()
    # new_selection = autos_selection()
    return 1


def main():
    optimal_pruning_ratio = choose_clients()
    print('================================finish========================================')
    
if __name__ == "__main__":

    main()
