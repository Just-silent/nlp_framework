# -*- coding: utf-8 -*-
# @Author   : Just-silent
# @time     : 2020/9/18 8:17

import math
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import *
from common.model.common_model import CommonModel

RANDOM_SEED = 2020

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


class ODModel(CommonModel):
    '''
    Transformer Parameters
        src – the sequence to the encoder (required).                                       (N, S, E)
        tgt – the sequence to the decoder (required).                                       (N, T, E)
        src_mask – the additive mask for the src sequence (optional).                       (S,S)
        tgt_mask – the additive mask for the tgt sequence (optional).                       (T,T)
        memory_mask – the additive mask for the encoder output (optional).                  (T,S)
        src_key_padding_mask – the ByteTensor mask for src keys per batch (optional).       (N,S)
        tgt_key_padding_mask – the ByteTensor mask for tgt keys per batch (optional).       (N,T)
        memory_key_padding_mask – the ByteTensor mask for memory keys per batch (optional). (N,S)
    '''
    def __init__(self, seq_config_file):
        super(ODModel, self).__init__(seq_config_file)
        self._config = seq_config_file
        self.bert = BertModel(self._config)
        self.embedding = nn.Embedding(self._config.data.num_vocab, self._config.model.dim_embedding)
        self.lstm_encoder = nn.LSTM(self._config.model.dim_embedding, self._config.model.dim_hidden // 2,
                                    num_layers=self._config.model.num_layer,
                                    bidirectional=self._config.model.bidirectional,
                                    batch_first=True)
        self.linner = nn.Linear(self._config.model.dim_hidden, self._config.data.num_vocab)
        self.pos_encoder = PositionalEncoding(self._config.model.dim_embedding, self._config.model.dropout)
        self.transformer = nn.Transformer(nhead=self._config.model.nhead,
                                          num_encoder_layers=self._config.model.num_encoder_layers)

    def generate_square_subsequent_mask(self, sz):
        # decoder self-att mask
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def get_attn_pad_mask(self, seq_q, seq_k):
        '''
        seq_q: [batch_size, seq_len]
        seq_k: [batch_size, seq_len]
        seq_len could be src_len or it could be tgt_len
        seq_len in seq_q and seq_len in seq_k maybe not equal
        '''
        batch_size, len_q = seq_q.size()
        batch_size, len_k = seq_k.size()
        # eq(zero) is PAD token
        pad_attn_mask = seq_k.data.eq(1).unsqueeze(1)  # [batch_size, 1, len_k], True is masked
        return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]

    def get_pad_mask(self, input):
        # eq(zero) is PAD token
        pad_attn_mask = input.data.eq(1)  # [batch_size, 1, len_k], True is masked
        return pad_attn_mask

    def get_attn_subsequence_mask(self, seq):
        '''
        seq: [batch_size, tgt_len]
        '''
        attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
        subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # Upper triangular matrix
        subsequence_mask = torch.from_numpy(subsequence_mask).byte()
        return subsequence_mask.to(self._config.device)  # [batch_size, tgt_len, tgt_len]

    def get_mask(self, seq):
        mask = (torch.triu(torch.ones(seq.size(1), seq.size(1))) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(self._config.device)

    def forward(self, dict_inputs: dict) -> dict:
        dict_outputs = {'enc_seq_len': dict_inputs['enc_input'][1],
                        'dec_seq_len': dict_inputs['dec_input'][1]}
        enc_input = dict_inputs['enc_input'][0]
        dec_input = dict_inputs['dec_input'][0]
        tag = dict_inputs['tag']
        enc_embedding = self.pos_encoder(self.embedding(enc_input))
        dec_embedding = self.pos_encoder(self.embedding(dec_input))
        enc_self_attn_pad_mask = self.get_pad_mask(enc_input)
        dec_self_attn_pad_mask = self.get_pad_mask(dec_input)
        dec_self_attn_mask = self.get_mask(dec_input)
        memory_mask = None
        hidden = self.transformer(enc_embedding.transpose(0, 1), dec_embedding.transpose(0, 1), tgt_mask=dec_self_attn_mask,
                             memory_mask=memory_mask, src_key_padding_mask=enc_self_attn_pad_mask,
                             tgt_key_padding_mask=dec_self_attn_pad_mask,
                             memory_key_padding_mask=enc_self_attn_pad_mask).transpose(0, 1)
        emissions = self.linner(hidden)
        dict_outputs['emissions'] = emissions
        dict_outputs['target_sequence'] = tag
        if self.training:
            dict_outputs['loss_batch'] = F.cross_entropy(emissions.view(-1, self._config.data.num_vocab), tag.view(-1))
        else:
            dict_outputs['outputs'] = emissions.argmax(dim=-1)
        return dict_outputs
        pass


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
