# -*- coding: utf-8 -*-
# @Author   : Just-silent
# @time     : 2020/9/18 8:17

import random
import torch
import torch.nn as nn

from torchcrf import CRF

from common.model.common_model import CommonModel

RANDOM_SEED = 2020

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


class EventExteactModel(CommonModel):
    def __init__(self, seq_config_file):
        super(EventExteactModel, self).__init__(seq_config_file)
        self._config = seq_config_file
        self.embedding = nn.Embedding(self._config.data.num_vocab, self._config.model.dim_embedding)
        self.lstm_encoder = nn.LSTM(self._config.model.dim_embedding, self._config.model.dim_hidden // 2,
                                    num_layers=self._config.model.num_layer, bidirectional=self._config.model.bidirectional)
        self.crflayer = CRF(self._config.data.num_tag, batch_first=self._config.data.batch_first)
        self.linner = nn.Linear(self._config.model.dim_hidden, self._config.data.num_tag)

    def forward(self, dict_inputs: dict) -> dict:
        dict_outputs = {}
        dict_outputs['seq_len'] = dict_inputs['text'][1]
        text = dict_inputs['text'][0]
        tag = dict_inputs['tag']
        text_embedding = self.embedding(text)
        encoded, _ = self.lstm_encoder(text_embedding)
        emissions = self.linner(encoded)
        dict_outputs['emissions'] = emissions
        dict_outputs['target_sequence'] = tag
        if self.training:
            dict_outputs['loss_batch'] = -self.crflayer(emissions, tag)
        else:
            dict_outputs['outputs'] = self.crflayer.decode(emissions)
        return dict_outputs
        pass