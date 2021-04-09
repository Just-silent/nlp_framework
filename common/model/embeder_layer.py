'''
Author: WEY
Date: 2020-12-26 20:55:24
LastEditTime: 2021-01-12 00:11:41
'''
import torch
import os
import torch.nn as nn
import numpy as np
from allennlp.modules.elmo import Elmo, batch_to_ids
from pytorch_pretrained_bert import BertTokenizer, BertModel
from common.util.log import logger
from common.model.common_model import CommonModel
import torch.nn.functional as F
class EmbedLayer(CommonModel):
    def __init__(self, config):
        super(EmbedLayer, self).__init__(model_config = config)
        # build word representation
        self._device = config.device
        self._pretrain = config.model.pretrain
        self._elmo_options_file = os.path.join(config.model.elmo_model_dir, 'elmo_2x4096_512_2048cnn_2xhighway_options.json')
        self._elmo_weight_file = os.path.join(config.model.elmo_model_dir, 'elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5')
        self._use_char = config.model.use_char
        self._batch_size = config.data.batch_size
        self._char_hidden_dim = 0
        self._char_num_vocab = config.data.char_num_vocab
        self._char_all_feature = False
        if self._use_char:
            self._char_hidden_dim = config.model.char_hidden_dim
            self._char_embedding_dim = config.model.char_emb_dim
            if config.model.char_feature_extractor == 'CNN':
                self._char_feature = CharCNN(self._char_num_vocab, self._char_embedding_dim,
                                             self._char_hidden_dim, config.model.dropout)
            elif config.model.char_feature_extractor == 'LSTM':
                self._char_feature = CharBiLSTM(self._char_num_vocab, self._char_embedding_dim,
                                                self._char_hidden_dim, config.model.dropout, self._device)
            elif config.model.char_feature_extractor == 'GRU':
                self._char_feature = CharBiGRU(self._char_num_vocab, self._char_embedding_dim,
                                               self._char_hidden_dim, config.model.dropout, self._device)
            elif config.model.char_feature_extractor == 'ALL':
                self._char_all_feature = True
                self._char_feature = CharCNN(self._char_num_vocab, self._char_embedding_dim,
                                             self._char_hidden_dim, config.model.dropout)
                self._char_feature_extra = CharBiLSTM(self._char_num_vocab, self._char_embedding_dim,
                                                      self._char_hidden_dim, config.model.dropout, self._device)
            else:
                logger.error('Error char feature selection, '
                             'please check parameter config.char_feature_extractor (CNN/LSTM/GRU/ALL).')
                exit(0)
        self._embedding_dim = config.model.word_emb_dim
        self._word_num_vocab = config.data.word_num_vocab
        self._word_embedding = nn.Embedding(self._word_num_vocab, self._embedding_dim)
        if config.model.pretrain_word_embedding is not None:
            self._word_embedding.weight.data.copy_(torch.from_numpy(config.pretrain_word_embedding))
        else:
            self._word_embedding.weight.data.copy_(torch.from_numpy(self._random_embedding(self._word_num_vocab)))

        if self._pretrain == 'ELMo':
            self._elmo = Elmo(self._elmo_options_file, self._elmo_weight_file, 2, requires_grad=False, dropout=0)
        if self._pretrain == 'BERT':  # bert
            self._bert_tokenizer = BertTokenizer.from_pretrained(self._bert_model_dir, do_lower_case=False)
            self._bert_model = BertModel.from_pretrained(self._bert_model_dir)

        self._dropout = nn.Dropout(config.model.dropout)

    def _random_embedding(self, vocab_size):
        pretrain_emb = np.empty([vocab_size, self._embedding_dim])
        scale = np.sqrt(3.0 / self._embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, self._embedding_dim])
        return pretrain_emb

    def _bert_emb(self, batch_word_list, max_sent_len, select='first'):
        batch_emb = []
        for sent in batch_word_list:
            tokens = []
            slice = []
            tokens.append('[CLS]')
            for word in sent:
                token_set = self._bert_tokenizer.tokenize(word)
                if select == 'first':
                    slice.append(len(tokens))
                tokens.extend(token_set)
            tokens.append('[SEP]')
            if select == 'first':
                slice += list(range(len(tokens), len(tokens) + max_sent_len - len(sent)))
            else:
                slice += [[idx] for idx in range(len(tokens), len(tokens) + max_sent_len - len(sent))]
            tokens += ['[PAD]'] * (max_sent_len - len(sent))
            assert (len(slice) == max_sent_len)
            input_ids = self._bert_tokenizer.convert_tokens_to_ids(tokens)
            input_ids = torch.tensor([input_ids]).to(self._device)
            with torch.no_grad():
                atten_mask = input_ids.gt(0)
                bert_encode, _ = self._bert_model(input_ids, attention_mask=atten_mask)
                last_hidden = bert_encode[-1]
                if select == 'first':
                    slice = torch.tensor(slice).to(self._device)
                    sent_emb = torch.index_select(last_hidden, 1, slice)
                else:
                    sent_emb = []
                    for slice_token in slice:
                        slice_token = torch.tensor(slice_token).to(self._device)
                        token_emb = torch.mean(torch.index_select(last_hidden, 1, slice_token), dim=1)
                        sent_emb.append(token_emb)
                    sent_emb = torch.cat(sent_emb, dim=0).unsqueeze(0)
                batch_emb.append(sent_emb)

        batch_emb = torch.cat(batch_emb, 0)
        return batch_emb

    def forward(self, dict_inputs: dict)->dict:
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
        # original_words_batch = dict_inputs['sentence'] # bath size sentences
        word_inputs = dict_inputs['batch_sentences']
        char_len = dict_inputs['batch_char'].size(2) # max_char_len
        char_inputs = dict_inputs['batch_char'].view(-1,char_len) 
        char_seq_lengths = dict_inputs['batch_char_lengths'].view(-1)
        # char_seq_recover = dict_inputs['bach_char_recover']
        sent_len = word_inputs.size(0)
        batch_size = word_inputs.size(1)
        
        word_embs = self._word_embedding(word_inputs)
        # if self._pretrain == 'ELMo':
        #     character_ids = batch_to_ids(original_words_batch)
        #     character_ids = character_ids.to(self._device)
        #     embeddings = self._elmo(character_ids)
        #     elmo_embs = embeddings['elmo_representations'][-1]
        # if self._pretrain == 'BERT':
        #     bert_embs = self._bert_emb(original_words_batch, sent_len)

        # if self._pretrain == 'ELMo':
        #     word_list = [torch.cat([elmo_embs, word_embs], -1)]
        # elif self._pretrain == 'BERT':
        #     word_list = [torch.cat([bert_embs, word_embs], -1)]
        # else:
        word_list = [word_embs]

        if self._use_char:
            char_features = self._char_feature.get_last_hiddens(char_inputs, char_seq_lengths.cpu().numpy())
            # char_features = char_features[char_seq_recover]
            char_features = char_features.view(sent_len, batch_size, -1)
            # concat word and char together
            word_list.append(char_features)
            word_embs = torch.cat([word_embs, char_features], 2)
            if self._char_all_feature:
                char_features_extra = self._char_feature_extra.get_last_hiddens(
                    char_inputs, char_seq_lengths.cpu().numpy()
                )
                char_features_extra = char_features_extra[char_seq_recover]
                char_features_extra = char_features_extra.view(batch_size, sent_len, -1)
                # concat word and char together
                word_list.append(char_features_extra)
        word_embs = torch.cat(word_list, 2)
        word_represent = self._dropout(word_embs)
        dict_outputs['word_represent'] = word_represent
        return dict_outputs
    pass
class CharCNN(nn.Module):
    def __init__(self, char_alphabet_size, char_emb_dim, char_hidden_dim, dropout):
        super(CharCNN, self).__init__()
        # build char sequence feature extractor: CNN
        self._embedding_dim = char_emb_dim
        self._vocab_size = char_alphabet_size
        self._hidden_dim = char_hidden_dim
        self._char_drop = nn.Dropout(dropout)
        self._char_embeddings = nn.Embedding(self._vocab_size, self._embedding_dim)
        self._char_embeddings.weight.data.copy_(torch.from_numpy(self._random_embedding()))
        self._char_cnn = nn.Conv1d(self._embedding_dim, self._hidden_dim, kernel_size=3, padding=1)

    def _random_embedding(self):
        pretrain_emb = np.empty([self._vocab_size, self._embedding_dim])
        scale = np.sqrt(3.0 / self._embedding_dim)
        for index in range(self._vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, self._embedding_dim])
        return pretrain_emb

    def get_last_hiddens(self, input, seq_lengths):
        """
            input:
                input: Variable(batch_size, word_length)
                seq_lengths: numpy array (batch_size,  1)
            output:
                Variable(batch_size, char_hidden_dim)
            Note it only accepts ordered (length) variable, length size is recorded in seq_lengths
        """
        batch_size = input.size(0)
        char_embeds = self._char_drop(self._char_embeddings(input))
        char_embeds = char_embeds.transpose(2, 1).contiguous()
        char_cnn_out = self._char_cnn(char_embeds)
        char_cnn_out = F.max_pool1d(char_cnn_out, char_cnn_out.size(2)).view(batch_size, -1)
        return char_cnn_out

    def _get_all_hiddens(self, input, seq_lengths):
        """
            input:
                input: Variable(batch_size,  word_length)
                seq_lengths: numpy array (batch_size,  1)
            output:
                Variable(batch_size, word_length, char_hidden_dim)
            Note it only accepts ordered (length) variable, length size is recorded in seq_lengths
        """
        batch_size = input.size(0)
        char_embeds = self._char_drop(self._char_embeddings(input))
        char_embeds = char_embeds.transpose(2, 1).contiguous()
        char_cnn_out = self._char_cnn(char_embeds).transpose(2, 1).contiguous()
        return char_cnn_out

    def forward(self, input, seq_lengths):
        return self._get_all_hiddens(input, seq_lengths)
class CharBiLSTM(nn.Module):
    def __init__(self, alphabet_size, embedding_dim, hidden_dim, dropout, device='cpu', bidirect_flag=True):
        super(CharBiLSTM, self).__init__()
        # build char sequence feature extractor: LSTM ...
        self._device = device
        self._vocab_size = alphabet_size
        self._embedding_dim = embedding_dim
        self._hidden_dim = hidden_dim
        if bidirect_flag:
            self._hidden_dim = hidden_dim // 2
        self._char_drop = nn.Dropout(dropout)
        self._char_embeddings = nn.Embedding(self._vocab_size, self._embedding_dim)
        self._char_embeddings.weight.data.copy_(torch.from_numpy(self._random_embedding()))
        self._char_lstm = nn.LSTM(self._embedding_dim, self.hidden_dim, num_layers=1, batch_first=True,
                                  bidirectional=bidirect_flag)
        self._char_drop = self._char_drop.to(self._device)
        self._char_embeddings = self._char_embeddings.to(self._device)
        self._char_lstm = self._char_lstm.to(self._device)

    def _random_embedding(self):
        pretrain_emb = np.empty([self._vocab_size, self._embedding_dim])
        scale = np.sqrt(3.0 / self._embedding_dim)
        for index in range(self._vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, self._embedding_dim])
        return pretrain_emb

    def get_last_hiddens(self, input, seq_lengths):
        """
            input:
                input: Variable(batch_size, word_length)
                seq_lengths: numpy array (batch_size,  1)
            output:
                Variable(batch_size, char_hidden_dim)
            Note it only accepts ordered (length) variable, length size is recorded in seq_lengths
        """
        batch_size = input.size(0)
        char_embeds = self._char_drop(self._char_embeddings(input))
        char_hidden = None
        pack_input = pack_padded_sequence(char_embeds, seq_lengths, True)
        char_rnn_out, char_hidden = self._char_lstm(pack_input, char_hidden)
        # char_hidden = (h_t, c_t)
        #  char_hidden[0] = h_t = (2, batch_size, lstm_dimension)
        # char_rnn_out, _ = pad_packed_sequence(char_rnn_out)
        return char_hidden[0].transpose(1, 0).contiguous().view(batch_size, -1)

    def _get_all_hiddens(self, input, seq_lengths):
        """
            input:
                input: Variable(batch_size,  word_length)
                seq_lengths: numpy array (batch_size,  1)
            output:
                Variable(batch_size, word_length, char_hidden_dim)
            Note it only accepts ordered (length) variable, length size is recorded in seq_lengths
        """
        batch_size = input.size(0)
        char_embeds = self._char_drop(self._char_embeddings(input))
        char_hidden = None
        pack_input = pack_padded_sequence(char_embeds, seq_lengths, True)
        char_rnn_out, char_hidden = self._char_lstm(pack_input, char_hidden)
        char_rnn_out, _ = pad_packed_sequence(char_rnn_out)
        return char_rnn_out.transpose(1, 0)

    def forward(self, input, seq_lengths):
        return self._get_all_hiddens(input, seq_lengths)
class CharBiGRU(nn.Module):
    def __init__(self, alphabet_size, embedding_dim, hidden_dim, dropout, device='cpu', bidirect_flag=True):
        super(CharBiGRU, self).__init__()
        # build char sequence feature extractor: GRU ...
        self._device = device
        self._vocab_size = alphabet_size
        self._embedding_dim = embedding_dim
        self._hidden_dim = hidden_dim
        if bidirect_flag:
            self._hidden_dim = hidden_dim // 2
        self._char_drop = nn.Dropout(dropout)
        self._char_embeddings = nn.Embedding(self._vocab_size, self._embedding_dim)
        self._char_embeddings.weight.data.copy_(torch.from_numpy(self._random_embedding()))
        self._char_lstm = nn.GRU(self._embedding_dim, self._hidden_dim, num_layers=1, batch_first=True,
                                 bidirectional=bidirect_flag)
        self._char_drop = self._char_drop.to(self._device)
        self._char_embeddings = self._char_embeddings.to(self._device)
        self._char_lstm = self._char_lstm.to(self._device)

    def _random_embedding(self):
        pretrain_emb = np.empty([self._vocab_size, self._embedding_dim])
        scale = np.sqrt(3.0 / self._embedding_dim)
        for index in range(self._vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, self._embedding_dim])
        return pretrain_emb

    def get_last_hiddens(self, input, seq_lengths):
        """
            input:
                input: Variable(batch_size, word_length)
                seq_lengths: numpy array (batch_size,  1)
            output:
                Variable(batch_size, char_hidden_dim)
            Note it only accepts ordered (length) variable, length size is recorded in seq_lengths
        """
        batch_size = input.size(0)
        char_embeds = self._char_drop(self._char_embeddings(input))
        char_hidden = None
        pack_input = pack_padded_sequence(char_embeds, seq_lengths, True)
        char_rnn_out, char_hidden = self._char_lstm(pack_input, char_hidden)
        # char_rnn_out, _ = pad_packed_sequence(char_rnn_out)
        return char_hidden.transpose(1, 0).contiguous().view(batch_size, -1)

    def _get_all_hiddens(self, input, seq_lengths):
        """
            input:
                input: Variable(batch_size,  word_length)
                seq_lengths: numpy array (batch_size,  1)
            output:
                Variable(batch_size, word_length, char_hidden_dim)
            Note it only accepts ordered (length) variable, length size is recorded in seq_lengths
        """
        batch_size = input.size(0)
        char_embeds = self._char_drop(self._char_embeddings(input))
        char_hidden = None
        pack_input = pack_padded_sequence(char_embeds, seq_lengths, True)
        char_rnn_out, char_hidden = self._char_lstm(pack_input, char_hidden)
        char_rnn_out, _ = pad_packed_sequence(char_rnn_out)
        return char_rnn_out.transpose(1, 0)

    def forward(self, input, seq_lengths):
        return self._get_all_hiddens(input, seq_lengths)

    def __init__(self, input_size, hidden_size, dropout, bidirectional=True):
        super(LSTM, self).__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._dropout = dropout
        self._bidirectional = bidirectional
        self._rnn_cell = LSTMCell(self._input_size, self._hidden_size, self._dropout)
        if self._bidirectional:
            self._rnn_cell_back = LSTMCell(self._input_size, self._hidden_size, self._dropout)

    def forward(self, inputs_emb, mask, hidden):
        batch_size, max_seq_len = inputs_emb.size(0), inputs_emb.size(1)
        inputs = inputs_emb.transpose(0, 1)  # (seq_length, batch_size, input_size)
        rnn_outputs = []
        state = hidden
        for t in range(max_seq_len):
            output, state = self._rnn_cell.forward(x=inputs[t], state=state)
            rnn_outputs.append(output)
        rnn_outputs = torch.stack(rnn_outputs, dim=0).transpose(0, 1)  # (batch_size, seq_length, hidden_size)
        if self._bidirectional:
            seq_list = list(reversed(list(range(max_seq_len))))
            batch_sent_len = mask.long().sum(1)
            index_b, index_f = [], []
            for sent_idx in range(batch_size):
                sent_len = batch_sent_len[sent_idx].item()
                index_b.append(list(range(sent_len, max_seq_len)) + list(range(sent_len)))
                index_f.append(list(range(max_seq_len - sent_len, max_seq_len)) + list(range(max_seq_len - sent_len)))
            index_b = torch.LongTensor(index_b).to(inputs_emb.device)
            index_f = torch.LongTensor(index_f).to(inputs_emb.device)
            inputs_emb = torch.gather(inputs_emb, dim=1, index=index_b[:, :, None].expand_as(inputs_emb))
            inputs_back = inputs_emb.transpose(0, 1)  # (seq_len, batch_size, input_size)
            rnn_outputs_back = []
            state = hidden
            for t in seq_list:
                output, state = self._rnn_cell_back.forward(x=inputs_back[t], state=state)
                rnn_outputs_back.append(output)
            rnn_outputs_back = list(reversed(rnn_outputs_back))
            rnn_outputs_back = torch.stack(rnn_outputs_back, dim=0).transpose(0,
                                                                              1)  # (batch_size, seq_length, hidden_size)
            rnn_outputs_back = torch.gather(rnn_outputs_back, dim=1,
                                            index=index_f[:, :, None].expand_as(rnn_outputs_back))
            rnn_outputs = torch.cat([rnn_outputs, rnn_outputs_back], dim=-1)

        return rnn_outputs
