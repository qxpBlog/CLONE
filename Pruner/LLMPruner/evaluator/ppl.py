import torch
import numpy as np
import time 
from tqdm import tqdm
from codecarbon import track_emissions,EmissionsTracker
from LLMPruner.datasets.ppl_dataset import get_loaders

def PPLMetric(model, tokenizer, datasets, seq_len=128, batch_size = 4, device="cuda"):
    metric = {}
    for dataset in datasets:
        _, test_loader = get_loaders(dataset, tokenizer, seq_len=seq_len, batch_size = batch_size)
        if dataset == 'wikitext2':
            tracker = EmissionsTracker()
            tracker.start()
            start_time = time.time()
            ppl = llama_eval(model, test_loader, device)
            end_time = time.time()
            tracker.stop()
            latency = end_time - start_time
            print(f"Model latency: {latency:.3f} seconds")
            print("=============================================================")
        else:
            ppl = llama_eval(model, test_loader, device)
        metric[dataset] = ppl
        print(metric)
    return metric

def test_latency_energy(model, tokenizer, datasets, seq_len=128, batch_size = 4, device="cuda"):
    metric = {}
    _, test_loader_wikitext2 = get_loaders(datasets[0], tokenizer, seq_len=seq_len, batch_size = batch_size)
    _, test_loader_ptb = get_loaders(datasets[1], tokenizer, seq_len=seq_len, batch_size = batch_size)
    tracker = EmissionsTracker()
    tracker.start()
    start_time = time.time()
    ppl1 = llama_eval(model, test_loader_wikitext2, device)
    ppl2 = llama_eval(model, test_loader_ptb, device)
    end_time = time.time()
    tracker.stop()
    latency = end_time - start_time
    print(f"Model latency: {latency:.3f} seconds")
    metric[datasets[0]] = ppl1
    metric[datasets[1]] = ppl2
    print("=============================================================")
    return metric

@torch.no_grad()
def llama_eval(model, test_lodaer, device):
    nlls = []
    n_samples = 0
    for batch in tqdm(test_lodaer):
        batch = batch.to(device)
        output = model(batch)
        lm_logits = output.logits
        
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1))
        nlls.append(loss)
    #print(torch.cat(nlls, dim=-1).mean())
    ppl = np.exp(torch.cat(nlls, dim=-1).mean().item())
    return ppl.item()