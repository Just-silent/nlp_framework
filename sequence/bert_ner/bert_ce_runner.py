# coding:UTF-8
# author    :Just_silent
# init time :2021/4/12 9:08
# file      :event_extract_runner.py
# IDE       :PyCharm
import pandas as pd
import torch
import random
import warnings
import numpy as np
import torch.optim as optim

from tqdm import tqdm
from tool import *
from common.util.utils import timeit
from sequence.bert_ner.utils import Tool
from torch.optim.lr_scheduler import StepLR
from common.runner.bert_common_runner import BertCommonRunner
from sequence.bert_ner.bert_ce_config import BertConfig
from sequence.bert_ner.bert_ce_data import BertDataLoader
from sequence.bert_ner.bert_ce_evaluator import EventExtractEvaluator
from sequence.bert_ner.bert_ce_model import BertForSequenceTagging
from sequence.bert_ner.bert_ce_loss import SequenceCRFLoss

RANDOM_SEED = 2020

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

warnings.filterwarnings("ignore")

class Bert_Runner(BertCommonRunner):

    def __init__(self, seq_config_file):
        super(Bert_Runner, self).__init__(seq_config_file)
        self._max_f1 = -1
        self._tool = Tool()
        pass

    @timeit
    def _build_config(self):
        bert_config = BertConfig(self._config_file)
        self._config = bert_config.load_config()
        pass

    @timeit
    def _build_data(self):
        self.dataloader = BertDataLoader(self._config)

        self.tag2idx = self.dataloader.tag2idx
        self.idx2tag = self.dataloader.idx2tag

        self._config.model.ntag = len(self.idx2tag)

        self.train_data = self.dataloader.load_train()
        self.valid_data = self.dataloader.load_valid()
        self.test_data = self.dataloader.load_test()

        self.max_seq_len = self._config.data.max_len
        pass

    @timeit
    def _build_model(self):
        self._model = BertForSequenceTagging.from_pretrained(self._config.pretrained_models.dir, num_labels=self._config.model.ntag).to(self._config.device)
        pass

    @timeit
    def _build_loss(self):
        self._loss = SequenceCRFLoss(self._config).to(self._config.device)
        pass

    @timeit
    def _build_optimizer(self):
        optimizer_grouped_parameters = []
        if self._config.pretrained_models.full_finetuning:
            param_optimizer = list(self._model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay': self._config.learn.weight_decay},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
            ]
        else:
            param_optimizer = list(self._model.classifier.named_parameters())
            optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer]}]
        self._optimizer = optim.AdamW(optimizer_grouped_parameters, lr=float(self._config.learn.learning_rate))
        self._scheduler = StepLR(self._optimizer, step_size=2000, gamma=0.1)

    @timeit
    def _build_evaluator(self):
        self._evaluator = EventExtractEvaluator(self._config, self.idx2tag)

    @timeit
    def _valid(self, episode, valid_log_writer):
        print("begin validating epoch {}...".format(episode + 1))
        # switch to evaluate mode
        self.training = False
        self._model.eval()
        valid_data_iterator = self.dataloader.data_iterator(self.valid_data)
        steps = self.valid_data['size'] // self._config.data.batch_size
        for i in tqdm(range(steps)):
            batch_data, batch_token_starts, batch_tags = next(valid_data_iterator)
            batch_masks = batch_data.gt(0)
            input = {}
            input['input_ids'] = batch_data
            input['labels'] = batch_tags
            input['attention_mask'] = batch_masks
            input['input_token_starts'] = batch_token_starts
            dict_outputs = self._model(input)
            # self._display_output(dict_outputs['outputs'])
            dict_outputs['target_sequence'] = batch_tags
            # send batch pred and target
            self._evaluator.evaluate(dict_outputs['outputs'], dict_outputs['target_sequence'])
        # get the result
        f1 = self._evaluator.get_eval_output()
        return f1
        pass

    @timeit
    def valid(self):
        print("begin validating...")
        self._load_checkpoint()
        self._model.eval()
        valid_data_iterator = self.dataloader.data_iterator(self.valid_data)
        steps = self.valid_data['size'] // self._config.data.batch_size
        for i in tqdm(range(steps)):
            batch_data, batch_token_starts, batch_tags = next(valid_data_iterator)
            batch_masks = batch_data.gt(0)
            input = {}
            input['input_ids'] = batch_data
            input['labels'] = batch_tags
            input['attention_mask'] = batch_masks
            input['input_token_starts'] = batch_token_starts
            dict_outputs = self._model(input)
            # self._display_output(dict_outputs['outputs'])
            dict_outputs['target_sequence'] = batch_tags
            # send batch pred and target
            self._evaluator.evaluate(dict_outputs['outputs'], dict_outputs['target_sequence'])
        # get the result
        f1 = self._evaluator.get_eval_output()
        return f1
        pass

    def _display_output(self, dict_outputs):
        batch_data = dict_outputs['batch_data']

        word_vocab = batch_data.dataset.fields['text'].vocab
        tag_vocab = batch_data.dataset.fields['tag'].vocab

        batch_input_sequence = dict_outputs['input_sequence'].T
        import numpy as np
        batch_output_sequence = np.asarray(dict_outputs['outputs']).T
        batch_target_sequence = dict_outputs['target_sequence'].T

        result_format = "{}\t{}\t{}\n"
        for input_sequence, output_sequence, target_sequence in zip(
                batch_input_sequence, batch_output_sequence, batch_target_sequence):
            this_result = ""
            for word, tag, target in zip(input_sequence, output_sequence, target_sequence):
                if word != "<pad>":
                    this_result += result_format.format(
                        word_vocab.itos[word], tag_vocab.itos[tag], tag_vocab.itos[target]
                    )
            print(this_result + '\n')
        pass

    def test(self):
        self._load_checkpoint()
        self._model.eval()
        test_data_iterator = self.dataloader.data_iterator(self.test_data)
        steps = self.test_data['size'] // self._config.data.batch_size
        for i in tqdm(range(steps)):
            batch_data, batch_token_starts, batch_tags = next(test_data_iterator)
            batch_masks = batch_data.gt(0)
            input = {}
            input['input_ids'] = batch_data
            input['labels'] = batch_tags
            input['attention_mask'] = batch_masks
            input['input_token_starts'] = batch_token_starts
            dict_outputs = self._model(input)
            # self._display_output(dict_outputs['outputs'])
            dict_outputs['target_sequence'] = batch_tags
            # send batch pred and target
            self._evaluator.evaluate(dict_outputs['outputs'], dict_outputs['target_sequence'])
        # get the result
        f1 = self._evaluator.get_eval_output()
        return f1
        pass

    def _display_result(self, episode):
        pass

    def predict_test(self):
        self._load_checkpoint()
        print('finished load ')
        self._model.eval()
        self.training = False
        texts = []
        pred_tags = []
        ori_tags = []
        f = open(self._config.data.test_path, 'r', encoding='utf-8')
        texts = f.readlines()
        new_texts = []
        for text in tqdm(texts):
            text = text.strip()
            text_list = ['[CLS]']
            text_list.extend([c for c in text])
            ids = self.dataloader.tokenizer.convert_tokens_to_ids(text_list)
            token_starts = [0]
            token_starts.extend([1]*len(text))
            batch_data = torch.tensor(ids, dtype=torch.long).unsqueeze(0).repeat(self._config.data.batch_size, 1).to(self._config.device)
            batch_token_starts = torch.tensor(token_starts, dtype=torch.long).unsqueeze(0).repeat(self._config.data.batch_size, 1).to(self._config.device)
            input1 = {}
            input1['input_ids'] = batch_data
            input1['labels'] = None
            input1['attention_mask'] =  batch_data.gt(0)
            input1['input_token_starts'] = batch_token_starts
            tag_list = None
            if self._config.device=='cpu':
                if self._config.model.is_crf:
                    tag_list = [self.idx2tag[idx] for idx in self._model(input1)['outputs'][0]]
                else:
                    tag_list = [self.idx2tag[idx] for idx in self._model(input1)['outputs'].numpy().tolist()[0]]
            else:
                if self._config.model.is_crf:
                    tag_list = [self.idx2tag[idx] for idx in self._model(input1)['outputs'][0]]
                else:
                    tag_list = [self.idx2tag[idx] for idx in self._model(input1)['outputs'].cpu().numpy().tolist()[0]]
            pred_words = self._tool.get_result_by_sentence_tag(text_list[1:], tag_list)
            print(tag_list)
            print(pred_words)
            result = get_defect(text, tag_list)
            result = ' '.join(result).strip()
            pred_tags.append(result)
            new_texts.append(text)
        data = pd.DataFrame({'问题描述': new_texts, '三级故障描述模型结果': pred_tags})
        data.to_csv(self._config.data.test_result_save_path, index=False, encoding='utf_8_sig')

    def predict_test_valid(self):
        self._load_checkpoint()
        print('finished load ')
        self._model.eval()
        self.training = False
        texts = []
        tags = []
        data = pd.read_excel(self._config.data.test_result_save_path)
        for i in range(data.shape[0]):
            text = data.iloc[i, 2]
            if text is not np.nan:
                text = text.strip()
                text_list = ['[CLS]']
                text_list.extend([c for c in text])
                ids = self.dataloader.tokenizer.convert_tokens_to_ids(text_list)
                token_starts = [0]
                token_starts.extend([1]*len(text))
                batch_data = torch.tensor(ids, dtype=torch.long).unsqueeze(0).repeat(self._config.data.batch_size, 1).to(self._config.device)
                batch_token_starts = torch.tensor(token_starts, dtype=torch.long).unsqueeze(0).repeat(self._config.data.batch_size, 1).to(self._config.device)
                input1 = {}
                input1['input_ids'] = batch_data
                input1['labels'] = None
                input1['attention_mask'] =  batch_data.gt(0)
                input1['input_token_starts'] = batch_token_starts
                tag_list = None
                if self._config.device=='cpu':
                    if self._config.model.is_crf:
                        tag_list = [self.idx2tag[idx] for idx in self._model(input1)['outputs'][0]]
                    else:
                        tag_list = [self.idx2tag[idx] for idx in self._model(input1)['outputs'].numpy().tolist()[0]]
                else:
                    if self._config.model.is_crf:
                        tag_list = [self.idx2tag[idx] for idx in self._model(input1)['outputs'][0]]
                    else:
                        tag_list = [self.idx2tag[idx] for idx in self._model(input1)['outputs'].cpu().numpy().tolist()[0]]
                pred_words = self._tool.get_result_by_sentence_tag(text_list[1:], tag_list)
                print(tag_list)
                print(pred_words)
                result = get_defect(text, tag_list)
                result = ' '.join(result).strip()
                data.iloc[i, 4] = result
        data.to_excel(self._config.data.test_result_save_path, index=False, encoding='utf_8_sig')


if __name__ == '__main__':
    config_file = 'bert_ce_config.yml'

    runner = Bert_Runner(config_file)
    # runner.train()
    # runner.valid()
    # runner.test()
    # runner.predict_test()
    runner.predict_test_valid()
    pass