import csv
import os
import time
from abc import ABC
from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from base.runner.base_runner import BaseRunner
from common.evaluation.common_sequence_evaluator import CommonSeqEvaluator
from common.util.log import logger
from common.util.utils import timeit


class BertCommonRunner(BaseRunner, ABC):
    """
    common implementation for runner
    """
    def __init__(self, config_file):
        super(BaseRunner, self).__init__()

        self._config = None
        self._config_file = config_file

        self._train_dataloader = None
        self._valid_dataloader = None
        self._test_dataloader = None

        self._model = None
        self._loss = None

        self._optimizer = None
        # 将train_fmt移到build前，若不适合，方便在build内进行更改
        self._train_fmt = "train: episode={:4d}, global_step={:4d},  " \
                          "batch_size={:4d}, batch_loss={:.4f}, elapsed={:.4f}"

        self._build()

        #   for global_step
        self.global_step = 0

        #   for log
        dir_log = self._config.learn.dir.log
        Path(dir_log).mkdir(parents=True, exist_ok=True)
        self._valid_log_fields = ""
        self._valid_log_filepath = os.path.join(
            dir_log, self._config.model.name + "_valid_log.csv")

        dir_saved = self._config.learn.dir.saved
        Path(dir_saved).mkdir(parents=True, exist_ok=True)
        self._model_path = os.path.join(
            dir_saved, str(self._config.model.name + '.ckp'))

    def _build(self):
        self._build_config()
        self._build_data()
        self._build_model()
        self._build_loss()
        self._build_optimizer()
        self._build_evaluator()
        self._build_summaries()
        pass

    @timeit
    def _build_optimizer(self):
        if self._config.learn.optimizer.lower() == 'sgd':
            self._optimizer = optim.SGD(
                self._model.parameters(), lr=self._config.learn.learning_rate,
                momentum=self._config.learn.momentum, weight_decay=self._config.learn.weight_decay
            )
        elif self._config.learn.optimizer.lower() == 'adagrad':
            self._optimizer = optim.Adagrad(
                self._model.parameters(), lr=self._config.learn.learning_rate,
                weight_decay=self._config.learn.weight_decay
            )
        elif self._config.learn.optimizer.lower() == 'adadelta':
            self._optimizer = optim.Adadelta(
                self._model.parameters(), lr=self._config.learn.learning_rate,
                weight_decay=self._config.learn.weight_decay
            )
        elif self._config.learn.optimizer.lower() == 'rmsprop':
            self._optimizer = optim.RMSprop(
                self._model.parameters(), lr=self._config.learn.learning_rate,
                weight_decay=self._config.learn.weight_decay
            )
        elif self._config.learn.optimizer.lower() == 'adam':
            self._optimizer = optim.Adam(
                self._model.parameters(), lr=self._config.learn.learning_rate,
                weight_decay=self._config.learn.weight_decay
            )
        else:
            logger.error('Optimizer illegal: {}'.format(self._config.learn.optimizer))

        self._scheduler = StepLR(self._optimizer, step_size=2000, gamma=0.1)
        pass

    @timeit
    def _build_summaries(self):
        # for summary
        self._model_name = self._config.model.name + "_" + self._config.data.name
        dir_summary = self._config.learn.dir.summary
        Path(dir_summary).mkdir(parents=True, exist_ok=True)
        summary_log_dir = os.path.join(dir_summary, self._model_name)

        time = datetime.now().strftime("%Y%m%d%H%M%S")
        suffix = time[4:8] + "_" + time[8:12]
        # name = self._config.data.name + "-" + self._config.model.name + "-" + suffix
        summary_dir = summary_log_dir + '-' + suffix
        self._summary_writer = SummaryWriter(summary_dir)
        pass

    @timeit
    def _build_evaluator(self):
        self._evaluator = CommonSeqEvaluator()
        pass

    def _write_summary(self, step: int, result: dict, predix="train"):
        for result_name, result_value in result.items():
            self._summary_writer.add_scalar(
                predix + "/" + result_name, result_value, global_step=step)

    def train(self):
        # switch to train mode
        self._model.train()
        print("training...")
        f_max = 0.0
        with open(self._valid_log_filepath, mode='w') as valid_log_file:
            valid_log_writer = csv.writer(valid_log_file, delimiter=',')
            valid_log_writer.writerow(self._valid_log_fields)
            for episode in range(self._config.learn.episode):
                self._train_epoch(episode)
                f_value = self._valid(episode, valid_log_writer)
                if f_value > f_max:
                    self._save_checkpoint(episode)
                    print("The best model has been saved and its score is {:.4f}".format(f_value))
                    f_max = f_value
                else:
                    print("The score of the best model is {:.4f}".format(f_max))
                # self._display_result(episode)
                self._scheduler.step()
        self._summary_writer.close()
        pass

    def _train_epoch(self, episode):
        train_data_iterator = self.dataloader.data_iterator(self.train_data)
        epoch_start = time.time()
        self._model.train()
        steps = self.train_data['size']//self._config.data.batch_size//100
        for i in tqdm(range(steps)):
            batch_data, batch_token_starts, batch_tags = next(train_data_iterator)
            batch_masks = batch_data.gt(0)
            input = {}
            input['input_ids'] = batch_data
            input['labels'] = batch_tags
            input['attention_mask'] =batch_masks
            input['input_token_starts'] = batch_token_starts
            dict_output = self._model(input)
            dict_output['target_sequence'] = batch_tags
            dict_loss = self._loss(dict_output)

            # Backward and optimize
            self._optimizer.zero_grad()  # clear gradients for this training step
            batch_loss = dict_loss['loss_batch']
            batch_loss.backward()  # back-propagation, compute gradients
            self._optimizer.step()  # apply gradients
            self.global_step += 1
            self._write_summary(step=self.global_step, result=dict_loss)
            if self.global_step % self._config.learn.batch_display == 0:
                # summary_writer.flush()
                elapsed = time.time() - epoch_start
                print(self._train_fmt.format(
                    episode + 1, self.global_step,
                    self._config.data.batch_size,
                    batch_loss.item(), elapsed
                ))
                epoch_start = time.time()
        pass

    def _save_checkpoint(self, epoch):
        # for checkpoint
        dir_saved = self._config.learn.dir.saved
        Path(dir_saved).mkdir(parents=True, exist_ok=True)
        self._model_path = os.path.join(
            dir_saved, str(self._config.model.name + '.ckp'))

        torch.save({
            # 'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self._model.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict()
        }, self._model_path)
        pass

    def _load_checkpoint(self):
        config = Path(self._model_path)
        if config.is_file():
            print("loading saved pretrained model from {}.".format(self._model_path))
            checkpoint = torch.load(self._model_path)
            self._model.load_state_dict(checkpoint['model_state_dict'])
            self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # epoch = checkpoint['epoch']
            # loss = checkpoint['loss']
            self._model.to(self._config.device)
        else:
            print("No model exists in {}.".format(self._model_path))
        pass

    def test(self):
        self._load_checkpoint()
        self._model.eval()
        for dict_input in tqdm(self._test_dataloader):
            dict_output = self._model(dict_input)
            self._display_output(dict_output)
            # send batch pred and target
            self._evaluator.evaluate(dict_output)
        # get the result
        result = self._evaluator.get_eval_output()

    pass
