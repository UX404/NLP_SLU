import json

from utils.vocab import Vocab, LabelVocab
from utils.word2vec import Word2vecUtils
from utils.evaluator import Evaluator

import jieba.posseg as pseg
import numpy as np

dummy_ex = {
            "utt_id": 0,
            "manual_transcript": "",
            "asr_1best": " ",
            "semantic": []
        }
seg_idx_list = json.load(open('seg_idx_list.json', 'r'))
print(f'seg_idx_list:{len(seg_idx_list)}')
seg_idx_dict = dict(zip(seg_idx_list, range(len(seg_idx_list))))
# seg_idx_dict = {'ag':0, 'a':1, 'ad':2, 'an':3,
#                 'b':4, 'c':5, 'dg':6, 'd':7,
#                 'e':8, 'f':9, 'g':10, 'h':11,
#                 'i':12, 'j':13, 'k':14, 'l':15,
#                 'm':16, 'ng':17, 'n':18, 'nr':19,
#                 'ns':20, 'nt':21, 'nz':22, 'o':23,
#                 'p':24, 'q':25, 'r':26, 's':27,
#                 'tg':28, 't':29, 'u':30, 'vg':31,
#                 'v':32, 'vd':33, 'vn':34, 'w':35,
#                 'x':36, 'y':37, 'z':38, 'un':39}
def to_1h(id, maxl):
    x = np.zeros(maxl)
    x[id] = 1
    return x

class Example():

    @classmethod
    def configuration(cls, root, train_path=None, word2vec_path=None):
        cls.evaluator = Evaluator()
        cls.word_vocab = Vocab(padding=True, unk=True, filepath=train_path)
        cls.word2vec = Word2vecUtils(word2vec_path)
        cls.label_vocab = LabelVocab(root)

    @classmethod
    def load_dataset(cls, data_path, pre_data=False):
        datas = json.load(open(data_path, 'r', encoding='utf-8'))
        examples = []

        if not pre_data:
            for data in datas:
                for utt in data:
                    ex = cls(utt)
                    examples.append(ex)

            return examples

        else:
            pre_examples = []
            for data in datas:
                for i, utt in enumerate(data):
                    ex = cls(utt)
                    examples.append(ex)
                    if i == 0:
                        pre_ex = cls(dummy_ex)
                        pre_examples.append(pre_ex)
                    else:
                        pre_ex = cls(data[i-1])
                        pre_examples.append(pre_ex)

            return examples, pre_examples


    def __init__(self, ex: dict):
        super(Example, self).__init__()
        self.ex = ex

        self.id = ex['utt_id']
        self.pred = []
        
        self.utt = ex['asr_1best']
        self.words =pseg.cut(self.utt)
        # self.one_hot = 0
        self.one_hot = []
        for w in self.words:
            for _ in range(len(w.word)):
                self.one_hot.append(to_1h(seg_idx_dict[w.flag] if w.flag in seg_idx_dict.keys() else seg_idx_dict['un'], len(seg_idx_dict)))
        # self.one_hot = np.array(self.one_hot)
        self.slot = {}
        assert 'semantic' in ex or 'pred' in ex
        if 'semantic' in ex:
            labels = ex['semantic']
        elif 'pred' in ex:
            labels = ex['pred']
        for label in labels:
            act_slot = f'{label[0]}-{label[1]}'
            if len(label) == 3:
                self.slot[act_slot] = label[2]
        self.tags = ['O'] * len(self.utt)
        for slot in self.slot:
            value = self.slot[slot]
            bidx = self.utt.find(value)
            if bidx != -1:
                self.tags[bidx: bidx + len(value)] = [f'I-{slot}'] * len(value)
                self.tags[bidx] = f'B-{slot}'
        self.slotvalue = [f'{slot}-{value}' for slot, value in self.slot.items()]
        self.input_idx = [Example.word_vocab[c] for c in self.utt]
        l = Example.label_vocab
        self.tag_id = [l.convert_tag_to_idx(tag) for tag in self.tags]
