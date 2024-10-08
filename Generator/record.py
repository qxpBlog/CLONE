from typing import List

import torch
import torch.nn.functional as F
import numpy


class Record(object):
    def __init__(self, operation, performance):
        if isinstance(operation, List):
            self.operation = numpy.array(operation)    # numpy一维列表
        elif isinstance(operation, torch.Tensor):
            self.operation = operation.numpy()
        else:
            assert isinstance(operation, numpy.ndarray)
            self.operation = operation
        self.performance = performance

    def get_permutated(self):
        pass

    def get_ordered(self):
        pass

    def repeat(self):
        pass

    def __eq__(self, other):
        if not isinstance(other, Record):
            return False
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        return str(self.operation).__hash__()


class SelectionRecord(Record):
    def __init__(self, operation, performance):    # operation应该是numpy一维列表
        super().__init__(operation, performance)
        self.max_size = operation.shape[0]     # 总可选择数量, 32：每一层的剪枝率


    def _get_ordered(self):
        # indice_select = torch.arange(0, self.max_size * 1000)[self.operation != 0]    # self.operation存0，1，0表示没选择，非零表示选择
        indice_select = torch.tensor([x for x in self.operation if x != 0])  
        return indice_select, torch.FloatTensor([self.performance])    # performance转化为一维向量

    # 重排列增广,就是把剪枝率大于0的decoder层的顺序打乱，生成num个不同顺序的数据，但是他们的performance相同只是顺序不同
    def get_permutated(self, num=25, padding=True, padding_value=-1):   
        ordered, performance = self._get_ordered()
        size = ordered.shape[0]    # 选择的数量
        shuffled_indices = torch.empty(num + 1, size) # 扩展成(num + 1) * size
        shuffled_indices[0] = ordered    # shuffled_indices第一个位置存的原始排序，即数字从小到大
        label = performance.unsqueeze(0).repeat(num + 1, 1) # 扩展成(num + 1) * 1
        for i in range(num):    # 增加重排列的数量
            shuffled_indices[i + 1] = ordered[torch.randperm(size)]
        if padding and size < self.max_size:    # shuffled_indices padding至总可选择数量
            shuffled_indices = F.pad(shuffled_indices, (0, (self.max_size - size)), 'constant', padding_value)
        return shuffled_indices, label

    def repeat(self, num=25, padding=True, padding_value=-1):    # 重复
        ordered, performance = self._get_ordered()
        size = ordered.shape[0]
        label = performance.unsqueeze(0).repeat(num + 1, 1)
        indices = ordered.unsqueeze(0).repeat(num + 1, 1)
        if padding and size < self.max_size:
            indices = F.pad(indices, (0, (self.max_size - size)), 'constant', padding_value)
        return indices, label


class RecordList(object):
    def __init__(self):
        self.r_list = set()

    def append(self, op, val):    # 增添选择和其对应的分数
        self.r_list.add(SelectionRecord(op, val))

    def __len__(self):
        return len(self.r_list)

    def generate(self, num=25, padding=True, padding_value=-1):    #对存储的所有record进行重排列，输出的两个都是二维矩阵
        results = []
        labels = []
        for record in self.r_list:
            result, label = record.get_permutated(num, padding, padding_value)
            results.append(result)
            labels.append(label)

        return torch.cat(results, 0), torch.cat(labels, 0)
