# -*- coding: utf-8 -*-
# @Author   : Just-silent
# @time     : 2020/9/18 8:17

import torch
import torch.nn as nn

from transformers import *

from torch.nn.utils.rnn import pad_sequence

from common.model.common_model import CrfDecoder

class TextSimilarity(BertPreTrainedModel):
	def __init__(self, config):
		super(TextSimilarity, self).__init__(config)
		self.num_labels = config.num_labels

		self.bert = BertModel(config)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)
		self.classifier = nn.Linear(config.hidden_size, 2)

		# self.init_weights()

	def forward(self, input):
		input_ids = input['input_ids']
		labels = input['labels']
		attention_mask = input['attention_mask']
		input_token_starts = input['input_token_starts']
		token_type_ids = None
		position_ids = None
		inputs_embeds = None
		head_mask = None
		outputs = self.bert(input_ids,
							attention_mask=attention_mask,
							token_type_ids=input_token_starts,
							position_ids=position_ids,
							head_mask=head_mask,
							inputs_embeds=inputs_embeds)
		sequence_output = outputs[1]
		sequence_output = self.dropout(sequence_output)
		logits = self.classifier(sequence_output)
		logits = torch.softmax(logits, dim=-1)
		loss = None
		outputs = {}
		if labels is not None:
			loss_fct = nn.CrossEntropyLoss()
			if self.training:
				loss = loss_fct(logits.view(-1, 2), labels.view(-1))

		outputs['loss_batch'] = loss
		outputs['emissions'] = logits
		outputs['outputs'] = torch.argmax(logits, dim=-1)
		outputs['mask'] = attention_mask
		return outputs
