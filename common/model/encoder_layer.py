'''
Author: WEY
Date: 2020-12-27 22:10:16
LastEditTime: 2021-01-12 00:20:51
'''
import torch.nn as nn
import torch
import torch.nn.functional as F
from common.model.common_model import CommonModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
class EncodeLayer(CommonModel):
   def __init__(self, config):
        super(EncodeLayer, self).__init__(config)
        self._config = config
        self._device = config.device
        self._batch_size = config.data.batch_size
        self._use_char = config.model.use_char
        self.is_bidirectional = config.model.is_bidirectional
        self._lstm_layer = config.model.lstm_layers
        self._lstm_dropout = nn.Dropout(config.model.dropout)
        self._input_size = config.model.word_emb_dim
        self._char_hidden_dim =  config.model.char_hidden_dim
        self._lstm_hidden = config.model.hidden_dim 
        self._char_feature_extractor = config.model.char_feature_extractor 
        self._word_feature_extractor = config.model.word_feature_extractor
        self._pretrian = config.model.pretrain
        self._cnn_layer = config.model.cnn_layer
        if self._pretrian == 'ELMo':  # elmo
            self._input_size = 1024 + self._input_size
        if self._pretrian == 'BERT':  # bert
            self._input_size = 768 + self._input_size  # bert-base
        if self._use_char:
            self._input_size += self._char_hidden_dim
            if self._char_feature_extractor  == 'ALL':
                self._input_size += self._char_hidden_dim
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        if self.is_bidirectional:
            self._lstm_hidden = self._lstm_hidden // 2
        if self._word_feature_extractor == 'LSTM':
            self._lstm = nn.LSTM(self._input_size, self._lstm_hidden, bidirectional=self.is_bidirectional,num_layers=self._lstm_layer, dropout=config.model.dropout)
        # elif self._word_feature_extractor == 'CNN':
        #     # cnn_hidden = config.HP_hidden_dim
        #     self._word2cnn = nn.Linear(self._input_size, config.model.char_hidden_dim)
        #     print('CNN layer: ', self._cnn_layer)
        #     self._cnn_list = nn.ModuleList()
        #     self._cnn_drop_list = nn.ModuleList()
        #     self._cnn_batchnorm_list = nn.ModuleList()
        #     kernel = 3
        #     pad_size = int((kernel - 1) / 2)
        #     for idx in range(self._cnn_layer):
        #         self._cnn_list.append(
        #             nn.Conv1d(config.HP_hidden_dim, config.HP_hidden_dim, kernel_size=kernel, padding=pad_size))
        #         self._cnn_drop_list.append(nn.Dropout(config.HP_dropout))
        #         self._cnn_batchnorm_list.append(nn.BatchNorm1d(config.HP_hidden_dim))
   def __init_hidden(self, batch_size=None): 
       h0 = torch.zeros(self._lstm_layer * 2, batch_size, self._lstm_hidden).to(self._device) 
       c0 = torch.zeros(self._lstm_layer * 2, batch_size, self._lstm_hidden).to(self._device) 
       return h0, c0 
   def forward(self, dict_inputs: dict ) -> dict:
            """
                input:
                    word_inputs: (batch_size, sent_len)
                    word_seq_lengths: list of batch_size, (batch_size,1)
                    char_inputs: (batch_size*sent_len, word_length)
                    char_seq_lengths: list of whole batch_size for char, (batch_size*sent_len, 1)
                    char_seq_recover: variable which records the char order information, used to recover char order
                output:
                    Variable(batch_size, sent_len, hidden_dim)
            """
            dict_outputs = dict_inputs
            embed = dict_inputs['word_represent']
            # word_inputs = dict_inputs['batch_sentences']
            # mask = dict_inputs['mask']

            # word_represent = self._word_representation(dict_inputs)
            # word_embs (batch_size, seq_len, embed_size)
            # if self._word_feature_extractor == 'CNN':
            #     batch_size = word_inputs.size(0)
            #     word_in = torch.tanh(self._word2cnn(word_represent)).transpose(2, 1).contiguous()
            #     for idx in range(self._cnn_layer):
            #         if idx == 0:
            #             cnn_feature = F.relu(self._cnn_list[idx](word_in))
            #         else:
            #             cnn_feature = F.relu(self._cnn_list[idx](cnn_feature))
            #         cnn_feature = self._cnn_drop_list[idx](cnn_feature)
            #         if batch_size > 1:
            #             cnn_feature = self._cnn_batchnorm_list[idx](cnn_feature)
            #     feature_out = cnn_feature.transpose(2, 1).contiguous()
            #     outputs = self._hidden2tag(feature_out)
            if self._word_feature_extractor == 'LSTM':  # lstm
                h0,c0 = self.__init_hidden(batch_size=len(dict_outputs['batch_sent_lengths']))
                embed_pack = pack_padded_sequence(input = embed,lengths = dict_outputs['batch_sent_lengths'])
                lstm_out,(hn,cn) = self._lstm(embed_pack,(h0,c0)) # 256,15,350
                lstm_out,batch_sent_lengths = pad_packed_sequence(lstm_out)  
                hidden_outputs = self._lstm_dropout(lstm_out)
                dict_outputs['lstm_out'] = hidden_outputs           
            return dict_outputs






