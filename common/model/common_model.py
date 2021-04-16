# coding:UTF-8
# author    :Just_silent
# init time :2021/4/16 11:00
# file      :common_model1.py
# IDE       :PyCharm

from abc import ABC
from base.model.base_model import BaseModel

import torch.nn as nn
from torch import Module
from torch.nn.utils.rnn import pad_sequence
from transformers.modeling_bert import BertModel, BertPreTrainedModel


class CommonModel(BaseModel, ABC):
    def __init__(self, model_config):
        super(CommonModel, self).__init__()
        self._config = model_config
        self._device = self._config.device
        pass


class BertForSequenceTaggingEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForSequenceTaggingEncoder, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(self, input):
        input_ids = input['input_ids']
        attention_mask = input['attention_mask']
        input_token_starts = input['input_token_starts']
        token_type_ids = None
        position_ids = None
        inputs_embeds = None
        head_mask = None
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)
        sequence_output = outputs[0]

        #### 'X' label Issue Start ####
        # obtain original token representations from sub_words representations (by selecting the first sub_word)
        origin_sequence_output = [
            layer[starts.nonzero().squeeze(1)]
            for layer, starts in zip(sequence_output, input_token_starts)]
        padded_sequence_output = pad_sequence(origin_sequence_output, batch_first=True)
        # print("padded_sequence_output", padded_sequence_output.shape)
        padded_sequence_output = self.dropout(padded_sequence_output)
        #### 'X' label Issue End ####
        encoded = padded_sequence_output
        return encoded


class LstmEncoder(Module):
    def __init__(self):
        super(LstmEncoder, self).__init__()
        pass

    def forward(self):
        pass


class CnnEncoder(Module):
    def __init__(self):
        super(CnnEncoder, self).__init__()

    def forward(self):
        pass


class TransformerEncoder(Module):
    def __init__(self):
        super(TransformerEncoder, self).__init__()
        pass

    def forward(self):
        pass


class TransformerDecoder(Module):
    def __init__(self):
        super(TransformerDecoder, self).__init__()

    def forward(self):
        pass


class CrfDecoder(Module):
    def __init__(self):
        super(CrfDecoder, self).__init__()

    def forward(self):
        pass