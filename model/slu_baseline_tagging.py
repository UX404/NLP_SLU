#coding=utf8
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils


class InverseMultiHeadAttention(nn.Module):
    def __init__(self, input_size, dropout_rate=0.5, num_heads=8):
        super(InverseMultiHeadAttention, self).__init__()
        self.num_heads = num_heads

        self.att_size = input_size // num_heads
        self.hidden_size = self.num_heads * self.att_size
        self.scale = self.att_size ** -0.5

        self.Wq = nn.Linear(input_size, self.hidden_size, bias=False)
        self.Wk = nn.Linear(input_size, self.hidden_size, bias=False)
        self.Wv = nn.Linear(input_size, self.hidden_size, bias=False)

        self.att_dropout = nn.Dropout(dropout_rate)

        self.output_layer = nn.Linear(self.hidden_size, input_size, bias=False)

    def forward(self, q, k, v):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        q = self.Wq(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.Wk(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.Wv(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q.mul_(self.scale)
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        x = 1 - torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x).squeeze(1)

        assert x.size() == orig_q_size, f'output size {x.size()} != input size {orig_q_size}'
        return x


class SLUTagging(nn.Module):

    def __init__(self, config):
        super(SLUTagging, self).__init__()
        self.config = config
        self.cell = config.encoder_cell
        self.word_embed = nn.Embedding(config.vocab_size, config.embed_size, padding_idx=0)
        self.rnn = getattr(nn, self.cell)(config.embed_size, config.hidden_size // 2, num_layers=config.num_layer, bidirectional=True, batch_first=True)
        self.dropout_layer = nn.Dropout(p=config.dropout)
        self.output_layer = TaggingFNNDecoder(config.hidden_size, config.num_tags, config.tag_pad_idx)
        self.freq_layer = FreqFNN(config.embed_size)

        self.attention = InverseMultiHeadAttention(config.hidden_size, config.dropout)
        self.attention_norm_1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attention_norm_2 = nn.LayerNorm(config.hidden_size, eps=1e-6)

    def forward(self, batch, pre_batch):
        tag_ids = batch.tag_ids
        tag_mask = batch.tag_mask
        input_ids = batch.input_ids
        input_freq = batch.input_freq
        lengths = batch.lengths

        pre_tag_ids = pre_batch.tag_ids
        pre_tag_mask = pre_batch.tag_mask
        pre_input_ids = pre_batch.input_ids
        pre_lengths = pre_batch.lengths

        embed = self.word_embed(input_ids)

        most_freq, freq_loss = self.freq_layer(embed, input_freq)

        mask = torch.ones_like(embed, requires_grad=False)
        mask[torch.arange(embed.shape[0]), most_freq] = 0
        mask[torch.rand(embed.shape[0]) > 0.02] = 1
        embed = embed * mask

        packed_inputs = rnn_utils.pack_padded_sequence(embed, lengths, batch_first=True, enforce_sorted=False)
        packed_rnn_out, h_t_c_t = self.rnn(packed_inputs)  # bsize x seqlen x dim
        rnn_out, unpacked_len = rnn_utils.pad_packed_sequence(packed_rnn_out, batch_first=True)

        pre_embed = self.word_embed(pre_input_ids)
        pre_packed_inputs = rnn_utils.pack_padded_sequence(pre_embed, lengths, batch_first=True, enforce_sorted=False)
        pre_packed_rnn_out, pre_h_t_c_t = self.rnn(pre_packed_inputs)  # bsize x seqlen x dim
        pre_rnn_out, pre_unpacked_len = rnn_utils.pad_packed_sequence(pre_packed_rnn_out, batch_first=True)
        
        norm_rnn_out = self.attention_norm_1(rnn_out)
        pre_rnn_out = self.attention_norm_2(pre_rnn_out)

        rnn_out = rnn_out + self.attention(pre_rnn_out, norm_rnn_out, norm_rnn_out)
        # rnn_out = self.attention_norm(rnn_out)

        hiddens = self.dropout_layer(rnn_out)
        tag_output = self.output_layer(hiddens, tag_mask, tag_ids)
        return tag_output[0], tag_output[1] + freq_loss

    def decode(self, label_vocab, batch, pre_batch):
        batch_size = len(batch)
        labels = batch.labels
        prob, loss = self.forward(batch, pre_batch)
        predictions = []
        for i in range(batch_size):
            pred = torch.argmax(prob[i], dim=-1).cpu().tolist()
            pred_tuple = []
            idx_buff, tag_buff, pred_tags = [], [], []
            pred = pred[:len(batch.utt[i])]
            for idx, tid in enumerate(pred):
                tag = label_vocab.convert_idx_to_tag(tid)
                pred_tags.append(tag)
                if (tag == 'O' or tag.startswith('B')) and len(tag_buff) > 0:
                    slot = '-'.join(tag_buff[0].split('-')[1:])
                    value = ''.join([batch.utt[i][j] for j in idx_buff])
                    idx_buff, tag_buff = [], []
                    pred_tuple.append(f'{slot}-{value}')
                    if tag.startswith('B'):
                        idx_buff.append(idx)
                        tag_buff.append(tag)
                elif tag.startswith('I') or tag.startswith('B'):
                    idx_buff.append(idx)
                    tag_buff.append(tag)
            if len(tag_buff) > 0:
                slot = '-'.join(tag_buff[0].split('-')[1:])
                value = ''.join([batch.utt[i][j] for j in idx_buff])
                pred_tuple.append(f'{slot}-{value}')
            predictions.append(pred_tuple)
        return predictions, labels, loss.cpu().item()


class TaggingFNNDecoder(nn.Module):

    def __init__(self, input_size, num_tags, pad_id):
        super(TaggingFNNDecoder, self).__init__()
        self.num_tags = num_tags
        self.output_layer = nn.Linear(input_size, num_tags)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, hiddens, mask, labels=None):
        logits = self.output_layer(hiddens)
        logits += (1 - mask).unsqueeze(-1).repeat(1, 1, self.num_tags) * -1e32
        prob = torch.softmax(logits, dim=-1)
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
            return prob, loss
        return prob

class FreqFNN(nn.Module):

    def __init__(self, embed_size):
        super(FreqFNN, self).__init__()
        self.freq_net = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, 1),
        )
        self.output_layer = nn.Softmax()
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, embed, freq=None):
        output = self.freq_net(embed).squeeze(-1)
        prob = torch.softmax(output, dim=-1)
        most_freq = prob.argmax(dim=1)
        if freq is not None:
            loss = self.loss_fct(prob, freq)
            return most_freq, loss
        return most_freq
