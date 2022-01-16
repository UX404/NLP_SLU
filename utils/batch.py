#-*- coding:utf-8 -*-
import torch
import numpy as np
from utils.example import seg_idx_dict

def from_example_list(args, ex_list, pre_ex_list, device='cpu', train=True):
    # ex_list = sorted(ex_list, key=lambda x: len(x.input_idx), reverse=True)
    batch = Batch(ex_list, device)
    pre_batch = Batch(pre_ex_list, device)
    pad_idx = args.pad_idx
    tag_pad_idx = args.tag_pad_idx

    batch.utt = [ex.utt for ex in ex_list]
    pre_batch.utt = [pre_ex.utt for pre_ex in pre_ex_list]

    input_lens = [len(ex.input_idx) for ex in ex_list]
    pre_input_lens = [len(pre_ex.input_idx) for pre_ex in pre_ex_list]

    max_len = max(max(input_lens), max(pre_input_lens))

    input_ids = [ex.input_idx + [pad_idx] * (max_len - len(ex.input_idx)) for ex in ex_list]
    input_freq = [ex.input_freq for ex in ex_list]
    pre_input_ids = [pre_ex.input_idx + [pad_idx] * (max_len - len(pre_ex.input_idx)) for pre_ex in pre_ex_list]

    ex_1h = []
    for ex in ex_list:
        # print(f'len-{len(ex.one_hot)}')
        # print(max_len - len(ex.input_idx))
        tmp = ex.one_hot + (max_len - len(ex.input_idx)) * [np.zeros(len(seg_idx_dict))]
        # ex.one_hot = np.array(ex.one_hot)
        ex_1h.append(tmp)
    batch.one_hot = np.array(ex_1h).astype(np.float32)
    pre_ex_1h = []
    for ex in pre_ex_list:
        tmp = ex.one_hot + (max_len - len(ex.input_idx)) * [np.zeros(len(seg_idx_dict))]
        # ex.one_hot = np.array(ex.one_hot)
        pre_ex_1h.append(tmp)
    pre_batch.one_hot = np.array(pre_ex_1h).astype(np.float32)

    # ex_1h = None
    # for ex in ex_list:
    #     # print(f'len-{len(ex.one_hot)}')
    #     # print(max_len - len(ex.input_idx))
    #     tmp = np.concatenate((np.array(ex.one_hot).astype(np.float32), np.zeros((max_len - len(ex.input_idx), len(seg_idx_dict))).astype(np.float32)),axis=0)
    #     # ex.one_hot = np.array(ex.one_hot)
    #     if ex_1h is None:
    #         ex_1h = tmp.reshape(1,tmp.shape[0], tmp.shape[1])
    #     else:
    #         # print(ex_1h.shape)
    #         # print(ex.one_hot.shape)
    #         ex_1h = np.concatenate((ex_1h, tmp.reshape(1,tmp.shape[0], tmp.shape[1])), axis=0)
    # batch.one_hot = ex_1h.astype(np.float32)
    # pre_ex_1h = None
    # for ex in pre_ex_list:
    #     tmp = np.concatenate((np.array(ex.one_hot).astype(np.float32), np.zeros((max_len - len(ex.input_idx), len(seg_idx_dict))).astype(np.float32)),axis=0)
    #     # ex.one_hot = np.array(ex.one_hot)
    #     if pre_ex_1h is None:
    #         pre_ex_1h = tmp.reshape(1,tmp.shape[0], tmp.shape[1])
    #     else:
    #         # print(ex_1h.shape)
    #         # print(ex.one_hot.shape)
    #         pre_ex_1h = np.concatenate((pre_ex_1h, tmp.reshape(1,tmp.shape[0], tmp.shape[1])), axis=0)
    # pre_batch.one_hot = pre_ex_1h.astype(np.float32)

    batch.input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
    batch.input_freq = torch.tensor(input_freq, dtype=torch.long, device=device)
    batch.lengths = input_lens

    pre_batch.input_ids = torch.tensor(pre_input_ids, dtype=torch.long, device=device)
    pre_batch.lengths = pre_input_lens

    if train:
        batch.labels = [ex.slotvalue for ex in ex_list]
        tag_lens = [len(ex.tag_id) for ex in ex_list]
        max_tag_lens = max(tag_lens)
        tag_ids = [ex.tag_id + [tag_pad_idx] * (max_tag_lens - len(ex.tag_id)) for ex in ex_list]
        tag_mask = [[1] * len(ex.tag_id) + [0] * (max_tag_lens - len(ex.tag_id)) for ex in ex_list]
        batch.tag_ids = torch.tensor(tag_ids, dtype=torch.long, device=device)
        batch.tag_mask = torch.tensor(tag_mask, dtype=torch.float, device=device)

        pre_batch.labels = [pre_ex.slotvalue for pre_ex in pre_ex_list]
        pre_tag_lens = [len(pre_ex.tag_id) for pre_ex in pre_ex_list]
        max_pre_tag_lens = max(pre_tag_lens)
        pre_tag_ids = [pre_ex.tag_id + [tag_pad_idx] * (max_pre_tag_lens - len(pre_ex.tag_id)) for pre_ex in pre_ex_list]
        pre_tag_mask = [[1] * len(pre_ex.tag_id) + [0] * (max_pre_tag_lens - len(pre_ex.tag_id)) for pre_ex in pre_ex_list]
        pre_batch.tag_ids = torch.tensor(pre_tag_ids, dtype=torch.long, device=device)
        pre_batch.tag_mask = torch.tensor(pre_tag_mask, dtype=torch.float, device=device)
    else:
        batch.labels = None
        batch.tag_ids = None
        batch.tag_mask = None

        pre_batch.labels = None
        pre_batch.tag_ids = None
        pre_batch.tag_mask = None

    return batch, pre_batch


class Batch():

    def __init__(self, examples, device):
        super(Batch, self).__init__()

        self.examples = examples
        self.device = device

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]