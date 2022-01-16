#coding=utf8
import sys, os, time, gc
import json
import numpy as np

install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)

from utils.args import init_args
from utils.initialization import *
from utils.example import Example
from utils.batch import from_example_list
from utils.vocab import PAD
from model.slu_baseline_tagging import SLUTagging

# initialization params, output path, logger, random seed and torch.device
args = init_args(sys.argv[1:])
set_random_seed(args.seed)
device = set_torch_device(args.device)
print("Initialization finished ...")
print("Random seed is set to %d" % (args.seed))
print("Use GPU with index %s" % (args.device) if args.device >= 0 else "Use CPU as target torch device")

train_path = os.path.join(args.dataroot, 'train.json')
Example.configuration(args.dataroot, train_path=train_path, word2vec_path=args.word2vec_path)

args.vocab_size = Example.word_vocab.vocab_size
args.pad_idx = Example.word_vocab[PAD]
args.num_tags = Example.label_vocab.num_tags
args.tag_pad_idx = Example.label_vocab.convert_tag_to_idx(PAD)

model_para_path = 'model.bin'
test_set_path = 'data/test_unlabelled.json'

test_dataset, test_pre_dataset = Example.load_dataset(test_set_path, pre_data=True)

model = SLUTagging(args).to(device)
model.load_state_dict(torch.load(model_para_path)['model'])
Example.word2vec.load_embeddings(model.word_embed, Example.word_vocab, device=device)

predictions, labels = [], []
total_loss, count = 0, 0

with torch.no_grad():
    for i in range(0, len(test_dataset), args.batch_size):
        cur_dataset = test_dataset[i: i + args.batch_size]
        cur_pre_dataset = test_pre_dataset[i: i + args.batch_size]
        current_batch, current_pre_batch = from_example_list(args, cur_dataset, cur_pre_dataset, device, train=True)
        pred, label, loss = model.decode(Example.label_vocab, current_batch, current_pre_batch)
        for j in range(len(pred)):
            idx = i + j
            for k in range(len(pred[j])):
                test_dataset[idx].pred.append(pred[j][k].split('-'))


out_list = []
utt_list = []
for i in range(len(test_dataset)):
    if test_dataset[i].id == 1 and len(utt_list) != 0:
        out_list.append(utt_list)
        utt_list = []
    tmp_dict = dict()
    tmp_dict['utt_id'] = test_dataset[i].id
    tmp_dict['asr_1best'] = test_dataset[i].utt
    tmp_dict['pred'] = test_dataset[i].pred
    utt_list.append(tmp_dict)
out_list.append(utt_list)
with open('test.json', 'w', encoding='utf-8') as f:
    json.dump(out_list, f, ensure_ascii=False, indent=4)