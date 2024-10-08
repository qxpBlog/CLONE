import torch 
from torch.utils.data import Dataset, DataLoader
import json

class MyDataset(Dataset):
    def __init__(self, json_file) -> None:
        super().__init__()
        with open(json_file, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data_item = self.data[idx]
        t = 1
        a, b = t, 1 - t  # ppl中wikitext2和ptb的权重
        ppl = torch.tensor(a * data_item['wikitext2'] + b * data_item['ptb'], dtype=torch.float32)
        latency = torch.tensor(data_item['latency'], dtype=torch.float32)
        energy = torch.tensor(data_item['energy'], dtype=torch.float32)
        pruning_ratio = torch.tensor(data_item['pruning_ratio'], dtype=torch.float32)
        # return {'pruning_ratio': pruning_ratio, 'ppl': ppl, 'latency':latency, 'energy': energy}
        return pruning_ratio, ppl, latency, energy

def get_dataloader(json_file: str, batch_size=7, shuffle=True):
    my_dataset = MyDataset(json_file)
    dataloader = DataLoader(my_dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader