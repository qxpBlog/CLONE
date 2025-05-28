"""
feature env
interactive with the actor critic for the state and state after action
"""
import os
from collections import namedtuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from record import RecordList
from utils.logger import error, info


base_path = '.' # Your path to Generator


class Evaluator(object):
    def __init__(self, task_type=None, dataset=None):
        self.records = RecordList()
        self.ds_size = 32000 # 开始符和结束符
        self.records_num = 0
    def __len__(self):    # 返回记录列表长度
        return len(self.records)
  
    # 记录新增的选择和性能
    def _store_history(self, choice, performance):
        self.records.append(choice, performance)
        self.records_num += 1

    # 在硬盘也存一下记录
    def _flush_history(self, choices, performances, is_permuted, num, padding):
        if is_permuted:
            flag_1 = 'augmented'
        else:
            flag_1 = 'original'
        if padding:
            flag_2 = 'padded'
        else:
            flag_2 = 'not_padded'
        torch.save(choices, f'{base_path}/history/choice.{flag_1}.{flag_2}.{num}.pt')
        info(f'save the choice to {base_path}/history/choice.pt')
        torch.save(performances, f'{base_path}/history/performance.{flag_1}.{flag_2}.{num}.pt')
        info(f'save the performance to {base_path}/history/performance.pt')

    def _check_path(self):
        if not os.path.exists(f'{base_path}/history'):
            os.mkdir(f'{base_path}/history')

    # 把列表里的记录按照增广和padding的形式存储到硬盘
    def save(self, num=25, padding=True, padding_value=-1):
        if num > 0:    # num为新增重拍列数量
            is_permuted = True
        else:
            is_permuted = False
        info('save the records...')
        choices, performances = \
            self.records.generate(num=num, padding=padding, padding_value=padding_value)    # 生成增广的重拍列
        self._flush_history(choices, performances, is_permuted, num, padding)

    def get_record(self, num=0, eos=-1):    # 输出存储的记录，以增广之后的形式，eos是填充的下标值
        results = []
        labels = []
        for record in self.records.r_list:
            result, label = record.get_permutated(num, True, eos)
            results.append(result)
            labels.append(label)
            # result, label = record.operation, record.performance
            # results.append(torch.Tensor(result).reshape(1, 32))
            # labels.append(torch.Tensor(label).reshape(1))
        return torch.cat(results, 0), torch.cat(labels, 0)



    def report_performance(self, choice, performances, store=True, rp=True, flag=''):    # 输出并记录（可选）对应选择的下游表现
        if store:    # 记录结果
            self._store_history(choice, performances)





