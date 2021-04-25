# -*- coding: utf-8 -*-
# @Author   : Just-silent
# @time     : 2020/9/18 8:17

from transformers.modeling_bert import *
from torch.nn.utils.rnn import pad_sequence

from common.model.common_model import CrfDecoder

class IntentionClassification(BertPreTrainedModel):
	def __init__(self, config):
		super(IntentionClassification, self).__init__(config)
		self.num_labels = config.num_labels

		self.bert = BertModel(config)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)
		self.classifier = nn.Linear(config.hidden_size, config.num_labels)
		self.crf_decoder = CrfDecoder(config.num_labels, batch_first=True)

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
							token_type_ids=token_type_ids,
							position_ids=position_ids,
							head_mask=head_mask,
							inputs_embeds=inputs_embeds)
		sequence_output = outputs[1]

		logits = self.classifier(sequence_output)

		loss_mask = None
		loss = None
		outputs = {}
		if labels is not None:
			# loss_mask = labels.gt(-1)
			loss_fct = CrossEntropyLoss()
			# # Only keep active parts of the loss
			# if loss_mask is not None:
			# 	active_loss = loss_mask.view(-1) == 1
			# 	active_logits = logits.view(-1, self.num_labels)[active_loss]
			# 	active_labels = labels.view(-1)[active_loss]
			# 	loss = loss_fct(active_logits, active_labels)
			# else:
			if self.training:
				loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

		outputs['loss_batch'] = loss
		outputs['emissions'] = logits
		outputs['outputs'] = torch.argmax(logits, dim=-1)
		outputs['mask'] = input_token_starts
		return outputs  # (loss), scores
