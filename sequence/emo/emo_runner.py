# coding:UTF-8
# author    :Just_silent
# init time :2021/4/12 9:08
# file      :emo_runner.py
# IDE       :PyCharm


import random, os
# os.environ['CUDA_ENABLE_DEVICES'] = '0'
import torch
import warnings
from tqdm import tqdm
import torch.optim as optim
from openpyxl import load_workbook
from torch.optim.lr_scheduler import StepLR

from sequence.event_extract.utils import Tool
from common.runner.common_runner import CommonRunner
from sequence.emo.emo_config import EmoConfig
from sequence.emo.emo_data import SequenceDataLoader
from sequence.emo.emo_evaluator import EmotEvaluator
from sequence.emo.emo_model import EmoModel
from sequence.emo.emo_loss import SequenceCRFLoss

RANDOM_SEED = 2020

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
warnings.filterwarnings("ignore")


class EmoRunner(CommonRunner):

    def __init__(self, seq_config_file):
        super(EmoRunner, self).__init__(seq_config_file)
        self._max_f1 = -1
        self._tool = Tool()
        pass

    def _build_config(self):
        flat_config = EmoConfig(self._config_file)
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
        self._test_dataloader = self._dataloader.load_test()

        train_max_len = max([len(example.text) for example in self._train_dataloader.dataset.examples])
        valid_max_len = max([len(example.text) for example in self._valid_dataloader.dataset.examples])
        self.max_seq_len = max(train_max_len, valid_max_len)
        pass

    def _build_model(self):
        self._model = EmoModel(self._config).to(self._config.device)
        pass

    def _build_loss(self):
        self._loss = SequenceCRFLoss(self._config).to(self._config.device)
        pass

    def _build_optimizer(self):
        self._optimizer = optim.SGD(self._model.parameters(), lr=float(self._config.learn.learning_rate),
                                    momentum=self._config.learn.momentum)
        self._scheduler = StepLR(self._optimizer, step_size=2000, gamma=0.1)

    def _build_evaluator(self):
        self._evaluator = EmotEvaluator(self._config, self.tag_vocab)

    def _valid(self, episode, valid_log_writer):
        print("begin validating epoch {}...".format(episode + 1))
        # switch to evaluate mode
        self._model.eval()
        for dict_input in tqdm(self._valid_dataloader):
            dict_input.training = False
            input = {'text': dict_input.text,
                     'tag': dict_input.tag}
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
            dict_input.training = False
            input = {'text': dict_input.text,
                     'tag': dict_input.tag}
            dict_outputs = self._model(input)
            # self._display_output(dict_outputs['outputs'])
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
            dict_input.training = False
            input = {'text': dict_input.text,
                     'tag': dict_input.tag}
            dict_outputs = self._model(input)
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
            for c in text.split():
                sentence.append(c)
                if c in self.word_vocab.stoi.keys():
                    text_list.append(self.word_vocab.stoi[c])
                else:
                    text_list.append(0)
            text = torch.IntTensor(text_list).to(self._config.device)
            text_len = torch.IntTensor([len(text_list)]).to(self._config.device)
            text = text.repeat(1, 1)
            text_len = text_len.repeat(1, 1)
            dict_input = {}
            dict_input['text'] = [text, text_len]
            dict_input['tag'] = None
            dict_output = self._model(dict_input)
            p_index = self.tag_vocab.itos[dict_output['outputs'][0]]
            max_p = max(dict_output['emissions'][0])
            print('预测结果：{}'.format(p_index), '该类别的概率：{}'.format(max_p))


if __name__ == '__main__':
    config_file = 'emo_config.yml'

    runner = EmoRunner(config_file)
    runner.train()
    runner.valid()
    runner.test()
    runner.predict_test()
    pass
