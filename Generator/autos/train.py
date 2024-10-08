import argparse
import os
import sys

import pandas

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import pickle
import random
import sys
from typing import List

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils
from torch import Tensor
from torch.utils.data import DataLoader

from model import AUTOS
from feature_env import Evaluator, base_path
from train_utils import AvgrageMeter, pairwise_accuracy, hamming_distance, FSDataset
from record import SelectionRecord
from utils.logger import info, error

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--new_gen', type=int, default=10)
parser.add_argument('--method_name', type=str, choices=['rnn'], default='rnn')
parser.add_argument('--gpu', type=int, default=0, help='used gpu')
parser.add_argument('--fe', type=str, choices=['+', '', '-'], default='-')
parser.add_argument('--top_k', type=int, default=20)
parser.add_argument('--gen_num', type=int, default=25)

parser.add_argument('--encoder_layers', type=int, default=1)
parser.add_argument('--encoder_hidden_size', type=int, default=256)
parser.add_argument('--encoder_emb_size', type=int, default=32000)
parser.add_argument('--mlp_layers', type=int, default=2)
parser.add_argument('--mlp_hidden_size', type=int, default=256)

parser.add_argument('--decoder_layers', type=int, default=1)
parser.add_argument('--decoder_hidden_size', type=int, default=256)
parser.add_argument('--encoder_dropout', type=float, default=0)
parser.add_argument('--mlp_dropout', type=float, default=0)
parser.add_argument('--decoder_dropout', type=float, default=0)

parser.add_argument('--l2_reg', type=float, default=0.0)
parser.add_argument('--max_step_size', type=int, default=100)
parser.add_argument('--trade_off', type=float, default=0.8)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--grad_bound', type=float, default=5.0)
parser.add_argument('--vocab_size', type=int, default=32000)
args = parser.parse_args()

baseline_name = [
    'kbest',
    'mrmr',
    'lasso',
    'rfe',
    # 'gfs',
    'lassonet',
    'sarlfs',
    'marlfs',
]


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6

def gains_train(train_queue, model: AUTOS, optimizer):
    objs = AvgrageMeter()
    mse = AvgrageMeter()
    nll = AvgrageMeter()
    model.train()
    for step, sample in enumerate(train_queue):
        encoder_input = sample['encoder_input']
        encoder_target = sample['encoder_target']
        decoder_input = sample['decoder_input']
        decoder_target = sample['decoder_target']
        encoder_input = encoder_input.cuda(model.gpu)
        encoder_target = encoder_target.cuda(model.gpu).requires_grad_()
        decoder_input = decoder_input.cuda(model.gpu)
        decoder_target = decoder_target.cuda(model.gpu)

        optimizer.zero_grad()
        predict_value, log_prob, arch = model.forward(encoder_input, decoder_input)
        loss_1 = F.mse_loss(predict_value.squeeze(), encoder_target.squeeze()) # mse loss
        loss_2 = F.nll_loss(log_prob.contiguous().view(-1, log_prob.size(-1)), decoder_target.view(-1)) # ce loss
        loss = args.trade_off * loss_1 + (1 - args.trade_off) * loss_2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_bound)
        optimizer.step()

        n = encoder_input.size(0)
        objs.update(loss.data, n)
        mse.update(loss_1.data, n)
        nll.update(loss_2.data, n)

    return objs.avg, mse.avg, nll.avg


def gains_valid(queue, model: AUTOS):
    pa = AvgrageMeter()
    hs = AvgrageMeter()
    mse = AvgrageMeter()
    with torch.no_grad():
        model.eval()
        for step, sample in enumerate(queue):
            encoder_input = sample['encoder_input']
            encoder_target = sample['encoder_target']
            decoder_target = sample['decoder_target']

            encoder_input = encoder_input.cuda(model.gpu)
            encoder_target = encoder_target.cuda(model.gpu)
            decoder_target = decoder_target.cuda(model.gpu)

            predict_value, logits, arch = model.forward(encoder_input)
            n = encoder_input.size(0)
            pairwise_acc = pairwise_accuracy(encoder_target.data.squeeze().tolist(),
                                             predict_value.data.squeeze().tolist())
            hamming_dis = hamming_distance(decoder_target.data.squeeze().tolist(), arch.data.squeeze().tolist())
            mse.update(F.mse_loss(predict_value.data.squeeze(), encoder_target.data.squeeze()), n)
            pa.update(pairwise_acc, n)
            hs.update(hamming_dis, n)
    return mse.avg, pa.avg, hs.avg


def choice_to_onehot(choice: List[int]):
    size = len(choice) * 1000
    onehot = torch.zeros(size + 1)
    onehot[torch.tensor(choice)] = 1
    return onehot[:-1]
    # if choice.dim() == 1:
    #     selected = torch.zeros_like(choice)
    #     selected[choice] = 1
    #     return selected[1:-1]
    # else:
    #     onehot = torch.empty_like(choice)
    #     for i in range(choice.shape[0]):
    #         onehot[i] = choice_to_onehot(choice[i])
    #     return onehot


def gains_infer(queue, model, step, direction='+'):
    new_gen_list = []
    model.eval()
    for i, sample in enumerate(queue):
        encoder_input = sample['encoder_input']
        encoder_input = encoder_input.cuda(model.gpu)
        model.zero_grad()
        new_gen = model.generate_new_feature(encoder_input, predict_lambda=step, direction=direction)
        new_gen_list.extend(new_gen.data.squeeze().tolist())
    return new_gen_list


def select_top_k(choice: Tensor, labels: Tensor, k: int) -> (Tensor, Tensor):
    values, indices = torch.topk(labels, k, dim=0)
    return choice[indices.squeeze()], labels[indices.squeeze()]


def autos_selection():
    if not torch.cuda.is_available():
        info('No GPU found!')
        sys.exit(1)
    # os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in args.gpu)
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.enabled = True
    cudnn.benchmark = False
    cudnn.deterministic = True
    device = int(args.gpu)
    info(f"Args = {args}")

    with open(f'{base_path}/history/pruning_ratio-performance.pkl', 'rb') as f:
        fe: Evaluator = pickle.load(f)
    model = AUTOS(args)
    record_list = [list(map(int, ls.operation)) for ls in fe.records.r_list]
    print(len(record_list))
    info(f"param size = {count_parameters_in_MB(model)}MB")
    model = model.cuda(device)

    
    choice, labels = fe.get_record(args.gen_num, eos=fe.ds_size)
    valid_choice, valid_labels = fe.get_record(0, eos=fe.ds_size)
    
    info('Training Encoder-Predictor-Decoder')

    min_val = min(labels)
    max_val = max(labels)
    # performance归一化
    train_encoder_target = [(i - min_val) / (max_val - min_val) for i in labels]
    valid_encoder_target = [(i - min_val) / (max_val - min_val) for i in valid_labels]
    
    # 加载数据
    train_dataset = FSDataset(choice, train_encoder_target, train=True, sos_id=fe.ds_size, eos_id=fe.ds_size)
    valid_dataset = FSDataset(valid_choice, valid_encoder_target, train=False, sos_id=fe.ds_size, eos_id=fe.ds_size)
    
    train_queue = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    valid_queue = torch.utils.data.DataLoader(
        valid_dataset, batch_size=len(valid_dataset), shuffle=False, pin_memory=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_reg)
    # 开始训练
    for epoch in range(1, args.epochs + 1):
        loss, mse, ce = gains_train(train_queue, model, optimizer)
        if epoch % 10 == 0 or epoch == 1:
            info("epoch {:04d} train loss {:.6f} mse {:.6f} ce {:.6f}".format(epoch, loss, mse, ce))
        if epoch % 100 == 0 or epoch == 1:
            mse, pa, hs = gains_valid(train_queue, model)
            info("Evaluation on train data")
            info('epoch {:04d} mse {:.6f} pairwise accuracy {:.6f} hamming distance {:.6f}'.format(epoch, mse, pa,
                                                                                                   hs))
            mse, pa, hs = gains_valid(valid_queue, model)
            info("Evaluation on valid data")
            info('epoch {:04d} mse {:.6f} pairwise accuracy {:.6f} hamming distance {:.6f}'.format(epoch, mse, pa,
                                                                                                   hs))

    top_selection, top_performance = select_top_k(valid_choice, valid_labels, args.top_k)

    infer_dataset = FSDataset(top_selection, top_performance, False, sos_id=fe.ds_size, eos_id=fe.ds_size)
    infer_queue = DataLoader(infer_dataset, batch_size=len(infer_dataset), shuffle=False,
                             pin_memory=True)
    new_selection = []
    new_selection_list = []
    new_choice = []
    predict_step_size = 0
    while len(new_selection) < args.new_gen:
        predict_step_size += 1
        info('Generate new architectures with step size {:d}'.format(predict_step_size))
        new_record = gains_infer(infer_queue, model, direction='+', step=predict_step_size)
        for choice in new_record:
            onehot_choice = choice_to_onehot(choice)
            if onehot_choice.sum() <= 0:
                error('insufficient selection')
                continue
            record = SelectionRecord(onehot_choice.numpy(), -1)
            new_list = [i for i, x in enumerate(record.operation) if x == 1]
            if new_list not in record_list and new_list not in new_selection_list and 6080 <= sum(map(lambda x : x % 1000, new_list)) <= 6720:
                new_selection.append(record)
                new_selection_list.append(new_list)
                new_choice.append(onehot_choice)
            if len(new_selection) >= args.new_gen:
                break
        info(f'{len(new_selection)} new choice generated now', )
        if predict_step_size > args.max_step_size:
            break
    info(f'build {len(new_selection)} new choice !!!')

    new_choice_pt = torch.stack(new_choice)
    if args.gen_num == 0:
        choice_path = f'{base_path}/history/generated_choice.pt'
    else:
        choice_path = f'{base_path}/history/generated_choice.pt'
    torch.save(new_choice_pt, choice_path)
    info(f'save generated choice to {choice_path}')

    # return new_selection
    # 保存模型参数
    torch.save(model.state_dict(), f'{base_path}/history/GenPruner.model_dict')
    for s in new_selection_list:
        # 获取值为1的元素的下标
        ans = [0.0] * 32
        for x in s:
            ans[x // 1000] = (x % 1000) / 1000
        # 打印结果
        # print(len(s))
        # print(s)
        print(ans)
        # print('[', end="")
        # for x in ans:
        #     print(x, end=" ")
        # print(']')
    exit(0)
    
    best_selection_test = None
    best_optimal_test = -1000
    for s in new_selection:
        test_data = fe.generate_data(s.operation, 'test')
        test_result = fe.get_performance(test_data)
        if test_result > best_optimal_test:
            best_selection_test = s.operation
            best_optimal_test = test_result
            info(f'found best on test : {best_optimal_test}')

    opt_path_test = f'{base_path}/history/{fe.task_name}/best-ours.hdf'
    test_p = fe.report_performance(best_selection_test, flag='test')
    info(f'found test generation in our method! the choice is {best_selection_test}, the performance is {test_p}')
    fe.generate_data(best_selection_test, 'train').to_hdf(opt_path_test, key='train')
    fe.generate_data(best_selection_test, 'test').to_hdf(opt_path_test, key='test')
    ps = []
    info('given overall validation')
    report_head = 'RAW\t'
    raw_test = pandas.read_hdf(f'{base_path}/history/{fe.task_name}.hdf', key='raw_test')
    ps.append('{:.2f}'.format(fe.get_performance(raw_test) * 100))
    for method in baseline_name:
        report_head += f'{method}\t'
        spe_test = pandas.read_hdf(f'{base_path}/history/{fe.task_name}.hdf', key=f'{method}_test')
        ps.append('{:.2f}'.format(fe.get_performance(spe_test) * 100))
    report_head += 'Ours\tOurs_Test'
    report = ''
    print(report_head)
    for per in ps:
        report += f'{per}&\t'
    report += '{:.2f}&\t'.format(test_p * 100)
    print(report)


if __name__ == '__main__':
    autos_selection()
