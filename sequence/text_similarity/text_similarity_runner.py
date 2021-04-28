# coding:UTF-8
# author    :Just_silent
# init time :2021/4/12 9:08
# file      :event_extract_runner.py
# IDE       :PyCharm


import torch
import random
import warnings
import numpy as np
import torch.optim as optim

from tqdm import tqdm
from common.util.utils import timeit
from torch.optim.lr_scheduler import StepLR
from common.runner.bert_common_runner import BertCommonRunner
from sequence.text_similarity.utils import Tool
from sequence.text_similarity.text_similarity_config import TextSimilarityConfig
from sequence.text_similarity.text_similarity_data import TextSimilarityDataLoader
from sequence.text_similarity.text_similarity_evaluator import TextSimilarityEvaluator
from sequence.text_similarity.text_similarity_model import TextSimilarity
from sequence.text_similarity.text_similarity_loss import SequenceCRFLoss

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
        bert_config = TextSimilarityConfig(self._config_file)
        self._config = bert_config.load_config()
        pass

    @timeit
    def _build_data(self):
        self.dataloader = TextSimilarityDataLoader(self._config)

        self.train_data = self.dataloader.load_train()
        self.valid_data = self.dataloader.load_valid()
        self.test_data = self.dataloader.load_test()

        self.max_seq_len = self._config.data.max_len
        pass

    @timeit
    def _build_model(self):
        self._model = TextSimilarity.from_pretrained(self._config.pretrained_models.dir).to(self._config.device)
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
        self._evaluator = TextSimilarityEvaluator(self._config)

    @timeit
    def _valid(self, episode, valid_log_writer):
        print("begin validating epoch {}...".format(episode + 1))
        # switch to evaluate mode
        self.training = False
        self._model.eval()
        valid_data_iterator = self.dataloader.data_iterator(self.valid_data)
        steps = self.valid_data['size'] // self._config.data.batch_size//100
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
        while True:
            print('请输入句子一：', end='')
            text1 = input()
            print('请输入句子二：', end='')
            text2 = input()
            text_list = ['[CLS]']
            text_list.extend([c for c in text1])
            text_list.extend('[SEP]')
            text_list.extend([c for c in text2])
            text_list.extend('[SEP]')
            ids = self.dataloader.tokenizer.convert_tokens_to_ids(text_list)
            token_starts = [0]*(len(text1)+2)
            token_starts.extend([1]*(len(text2)+1))
            batch_data = torch.tensor(ids, dtype=torch.long).unsqueeze(0).repeat(self._config.data.batch_size, 1).to(self._config.device)
            batch_token_starts = torch.tensor(token_starts, dtype=torch.long).unsqueeze(0).repeat(self._config.data.batch_size, 1).to(self._config.device)
            input1 = {}
            input1['input_ids'] = batch_data
            input1['labels'] = None
            input1['attention_mask'] =  batch_data.gt(0)
            input1['input_token_starts'] = batch_token_starts
            tag = None
            index = self._model(input1)['outputs'][0]
            if self._config.device=='cpu':
                tag = index.numpy().item()
            else:
                tag = index.cpu().numpy().item()
            print(tag)


if __name__ == '__main__':
    config_file = 'text_similarity_config.yml'

    runner = Bert_Runner(config_file)
    runner.train()
    runner.valid()
    runner.test()
    runner.predict_test()
    pass
