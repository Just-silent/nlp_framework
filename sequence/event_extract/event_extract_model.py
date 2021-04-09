# -*- coding: utf-8 -*-
# @Author   : Just-silent
# @time     : 2020/9/18 8:17

import csv
import time
import random
import torch
from tqdm import tqdm
from openpyxl import load_workbook
from tensorboardX import SummaryWriter

from sequence.ccks_ee.ccks_ee_evaluator import CCKS_Evaluator
from common.runner.common_runner import CommonRunner
from sequence.ccks_ee.ccks_ee_data import SequenceDataLoader
from sequence.ccks_ee.ccks_ee_loss import SequenceCRFLoss
from sequence.ccks_ee.ccks_ee_model import FLAT_changed
from sequence.ccks_ee.ccks_ee_config import CcksEeConfig

RANDOM_SEED = 2020

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


class CCKS_EE_Runner(CommonRunner):

    def __init__(self, seq_config_file):
        self._config = None
        super(CCKS_EE_Runner, self).__init__(seq_config_file)
        self._valid_log_fields = ['episode', 'P', 'R', 'F']
        #   for evaluator
        self._evaluator = CCKS_Evaluator(self._config, self.tag_vocab)
        self._max_f1 = -1
        pass

    def _build_config(self):
        ccks_ee_config = CcksEeConfig(self._config_file)
        self._config = ccks_ee_config.load_config()
        pass

    def _build_data(self):
        loader = SequenceDataLoader(self._config)

        self.word_vocab = loader.word_vocab
        self.entity_vocab = loader.entity_vocab
        self.tag_vocab = loader.tag_vocab

        self._config.data.num_vocab = len(self.word_vocab)
        self._config.data.num_tag = len(self.tag_vocab)

        self._train_dataloader = loader.load_train()
        self._valid_dataloader = loader.load_valid()
        self._test_dataloader = loader.load_test()
        pass

    def _build_model(self):
        self._model = FLAT_changed(self._config).to(self._config.device)
        pass

    def _build_loss(self):
        self._loss = SequenceCRFLoss(self._config)
        pass

    def train(self):
        # switch to train mode
        self._model.train()

        print("training...")
        with SummaryWriter(logdir=self._summary_log_dir, comment='model') as summary_writer, \
                open(self._valid_log_filepath, mode='w') as valid_log_file:
            valid_log_writer = csv.writer(valid_log_file, delimiter=',')
            valid_log_writer.writerow(self._valid_log_fields)
            for episode in range(self._config.learn.episode):
                self._train_epoch(episode, summary_writer)
                self._valid(episode, valid_log_writer, summary_writer)
                # self._display_result(episode)
                self._scheduler.step()
        pass

    def _train_epoch(self, episode, summary_writer):
        epoch_start = time.time()
        self._model.train()
        batch = 0
        for dict_input in tqdm(self._train_dataloader):

            dict_loss = self._model.loss(dict_input)
            # Backward and optimize
            self._optimizer.zero_grad()  # clear gradients for this training step
            batch_loss = dict_loss['loss_batch']
            batch_loss.backward()  # back-propagation, compute gradients
            self._optimizer.step()  # apply gradients

            self.global_step += 1
            batch += 1
            if self.global_step % self._config.learn.batch_display == 0:
                # for loss_key, loss_value in dict_loss.items():
                #     summary_writer.add_scalar('loss/' + loss_key, loss_value, self.global_step)
                # summary_writer.flush()
                elapsed = time.time() - epoch_start
                print(self._train_fmt.format(
                    episode + 1, self.global_step, batch,
                    self._config.data.train_batch_size,
                    batch_loss.item(), elapsed
                ))
                epoch_start = time.time()
        pass

    def valid(self):
        model = self._load_checkpoint()
        self._model.eval()
        for dict_input in tqdm(self._valid_dataloader):
            dict_outputs = self._model(dict_input)
            # self._display_output(dict_output)
            dict_outputs['target_sequence'] = dict_input.tag
            # send batch pred and target
            self._evaluator.evaluate(dict_outputs['outputs'], dict_outputs['target_sequence'].T)
        # get the result
        f1 = self._evaluator.get_eval_output()
        pass

    def _valid(self, episode, valid_log_writer, summary_writer):
        print("begin validating epoch {}...".format(episode + 1))
        # switch to evaluate mode
        self._model.eval()
        for dict_input in tqdm(self._valid_dataloader):
            dict_outputs = self._model(dict_input)
            # self._display_output(dict_outputs['outputs'])
            dict_outputs['target_sequence'] = dict_input.tag
            # send batch pred and target
            self._evaluator.evaluate(dict_outputs['outputs'], dict_outputs['target_sequence'].T)
        # get the result
        f1 = self._evaluator.get_eval_output()
        if self._max_f1 < f1:
            self._max_f1 = f1
            self._save_checkpoint(episode)
            pass
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
    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config_file = 'ccks_ee_config.yml'

    runner = CCKS_EE_Runner(config_file)
    runner.train()
    runner.valid()
    runner.predict_test()
    pass
