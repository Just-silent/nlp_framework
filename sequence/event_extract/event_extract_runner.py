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

from tensorboardX import SummaryWriter

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
        self._optimizer = optim.SGD(self._model.parameters(), lr=float(self._config.model.lr), momentum=self._config.model.momentum)


    def _valid(self, episode, valid_log_writer):
        print("begin validating epoch {}...".format(episode + 1))
        # switch to evaluate mode
        self._model.eval()
        for dict_input in tqdm(self._valid_dataloader):
            dict_input.training = False
            dict_outputs = self._model(dict_input, 0)
            # self._display_output(dict_outputs['outputs'])
            dict_outputs['target_sequence'] = dict_input.tag
            # send batch pred and target
            self._evaluator.evaluate(dict_outputs['outputs'], dict_outputs['target_sequence'],
                                     dict_outputs['seq_len'])
        # get the result
        f1 = self._evaluator.get_eval_output()
        if self._max_f1 < f1:
            self._max_f1 = f1
            self._save_checkpoint(episode)
            pass
        pass


    def valid(self):
        model = self._load_checkpoint()
        self._model.eval()
        for dict_input in tqdm(self._valid_dataloader):
            dict_outputs = self._model(dict_input)
            # self._display_output(dict_output)
            dict_outputs['target_sequence'] = dict_input.tag
            # send batch pred and target
            self._evaluator.evaluate(dict_outputs['outputs'], dict_outputs['target_sequence'],
                                     dict_outputs['seq_len'])
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
        model = self._load_checkpoint()
        self._model.eval()
        result_dict = {}
        for dict_input in tqdm(self._test_dataloader):
            ids = dict_input.id[0].to('cpu').tolist()
            entities_batch = dict_input.entity.transpose(0,1)
            dict_output = self._model(dict_input)
            for i, (pred, entities) in enumerate(zip(dict_output['outputs'], entities_batch)):
                result = {'origin_place':[],
                          'size':[],
                          'transfered_place':[]}
                entities_str = [self.entity_vocab.itos[index] for index in entities]
                pred_str = [self.tag_vocab.itos[index] for index in pred]
                for j in range(len(pred)):
                    if pred_str[j][2:] == 'origin_place':
                        result['origin_place'].append(entities_str[j])
                    elif pred_str[j][2:] == 'size':
                        result['size'].append(entities_str[j])
                    elif pred_str[j][2:] == 'transfered_place':
                        result['transfered_place'].append(entities_str[j])
                result_dict[ids[i]-1000] = result
        wb =  load_workbook(self._config.data.test_path)
        ws = wb['sheet1']
        for line, id_result in result_dict.items():
            ws.cell(line, 2, ','.join(id_result['origin_place']))
            ws.cell(line, 3, ','.join(id_result['size']))
            ws.cell(line, 4, ','.join(id_result['transfered_place']))
        wb.save(self._config.data.result_path)
            # self._display_output(dict_output)
            # send batch pred and target
            # self._metric.evaluate(dict_output['outputs'], dict_output['target_sequence'].T)
        # get the result
        # result = self._metric.get_eval_output()


if __name__ == '__main__':
    config_file = 'event_extract_config.yml'

    runner = FLAT_Runner(config_file)
    runner.train()
    runner.valid()
    runner.predict_test()
    pass