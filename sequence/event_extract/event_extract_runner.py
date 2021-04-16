# coding:UTF-8
# author    :Just_silent
# init time :2021/4/12 9:08
# file      :event_extract_runner.py
# IDE       :PyCharm


import random
import torch
import warnings
from tqdm import tqdm
import torch.optim as optim
from openpyxl import load_workbook

from torch.optim.lr_scheduler import StepLR

from sequence.event_extract.utils import Tool
from common.runner.common_runner import CommonRunner
from sequence.event_extract.event_extract_config import EventExtractConfig
from sequence.event_extract.event_extract_data import SequenceDataLoader
from sequence.event_extract.event_extract_evaluator import EventExtractEvaluator
from sequence.event_extract.event_extract_model import EventExteactModel
from sequence.event_extract.event_extract_loss import SequenceCRFLoss

RANDOM_SEED = 2020

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

warnings.filterwarnings("ignore")

class FLAT_Runner(CommonRunner):
    def __init__(self, seq_config_file):
        super(FLAT_Runner, self).__init__(seq_config_file)
        self._max_f1 = -1
        self._tool = Tool()
        pass

    def _build_config(self):
        flat_config = EventExtractConfig(self._config_file)
        self._config = flat_config.load_config()
        pass

    def _build_data(self):
        self._dataloader = SequenceDataLoader(self._config)

        self.word_vocab = self._dataloader.word_vocab
        self.tag_vocab = self._dataloader.tag_vocab
        self._config.model.ntag = len(self.tag_vocab.itos)

        self._config.data.num_vocab = len(self.word_vocab.itos)
        self._config.data.num_tag = len(self.tag_vocab.itos)

        self._train_dataloader = self._dataloader.load_train()
        self._valid_dataloader = self._dataloader.load_valid()

        train_max_len = max([len(example.tag) for example in self._train_dataloader.dataset.examples])
        valid_max_len = max([len(example.tag) for example in self._valid_dataloader.dataset.examples])
        self.max_seq_len = max(train_max_len, valid_max_len)
        pass

    def _build_model(self):
        self._model = EventExteactModel(self._config)
        pass

    def _build_loss(self):
        self._loss = SequenceCRFLoss(self._config).to(self._config.device)
        pass

    def _build_optimizer(self):
        self._optimizer = optim.SGD(self._model.parameters(), lr=float(self._config.learn.learning_rate), momentum=self._config.learn.momentum)
        self._scheduler = StepLR(self._optimizer, step_size=2000, gamma=0.1)

    def _build_evaluator(self):
        self._evaluator = EventExtractEvaluator(self._config, self.tag_vocab)

    def _valid(self, episode, valid_log_writer):
        print("begin validating epoch {}...".format(episode + 1))
        # switch to evaluate mode
        self._model.eval()
        for dict_input in tqdm(self._valid_dataloader):
            dict_input.training = False
            input = {}
            input['text'] = dict_input.text
            input['tag'] = dict_input.tag
            dict_outputs = self._model(input)
            # self._display_output(dict_outputs['outputs'])
            dict_outputs['target_sequence'] = dict_input.tag
            # send batch pred and target
            self._evaluator.evaluate(dict_outputs['outputs'], dict_outputs['target_sequence'])
        # get the result
        f1 = self._evaluator.get_eval_output()
        return f1
        pass

    def valid(self):
        self._load_checkpoint()
        self._model.eval()
        for dict_input in tqdm(self._valid_dataloader):
            input = {}
            input['text'] = dict_input.text
            input['tag'] = dict_input.tag
            dict_outputs = self._model(input)
            # self._display_output(dict_output)
            dict_outputs['target_sequence'] = dict_input.tag
            # send batch pred and target
            self._evaluator.evaluate(dict_outputs['outputs'], dict_outputs['target_sequence'])
        # get the result
        f1 = self._evaluator.get_eval_output()
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
        model = self._load_checkpoint()
        self._model.eval()
        for dict_input in tqdm(self._test_dataloader):
            dict_outputs = self._model(dict_input)
            # self._display_output(dict_output)
            dict_outputs['target_sequence'] = dict_input.tag
            # send batch pred and target
            self._evaluator.evaluate(dict_outputs['outputs'], dict_outputs['target_sequence'].T)
        # get the result
        f1 = self._evaluator.get_eval_output()

    def _display_result(self, episode):
        pass

    def predict_test(self):
        self._load_checkpoint()
        self._model.eval()
        result_dict = {}
        while True:
            print('请输入一段话：', end='')
            text = input()
            text_list = []
            sentence = []
            for c in text:
                sentence.append(c)
                if c in self.word_vocab.stoi.keys():
                    text_list.append(self.word_vocab.stoi[c])
                else:
                    text_list.append(0)
            text = torch.IntTensor(text_list, device=torch.device(self._config.device))
            text_len = torch.IntTensor([len(text_list)], device=torch.device(self._config.device))
            text = text.repeat(self._config.data.train_batch_size, 1)
            text_len = text_len.repeat(self._config.data.train_batch_size, 1)
            dict_input = {}
            dict_input['text'] = [text, text_len]
            dict_input['tag'] = None
            pred_tags = [self.tag_vocab.itos[index] for index in self._model(dict_input)['outputs'][0]]
            pred_words = self._tool.get_result_by_sentence_tag(sentence, pred_tags)
            print(pred_words)


if __name__ == '__main__':
    config_file = 'event_extract_config.yml'

    runner = FLAT_Runner(config_file)
    runner.train()
    runner.valid()
    runner.test()
    runner.predict_test()
    pass