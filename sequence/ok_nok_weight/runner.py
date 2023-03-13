# -*- coding:utf-8 _*-
"""
@version: 
author: just_silent
@time: 2022/02/25
@file: runner.py
@function: 
@modify: 
"""
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

from torch.nn import Module
import random

from sequence.ok_nok_weight.dataset import DataLoader


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(19980603)

class Model(Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linner = nn.Linear(10, 2)
        self.dropout = nn.Dropout(0.1)
        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax()
        pass

    def forward(self, input):
        dropout_output = self.dropout(input)
        output = self.linner(dropout_output)
        emission = self.softmax(output)
        return emission
        pass

    def loss(self, input, target):
        emission = self.forward(input)
        target = target
        loss = self.criterion(emission, target)
        return loss
        pass


class OkNok():
    def __init__(self):
        self.batch_size = 2
        self.model = Model()
        self.train_data_loader = None
        self.test_data_loader = None
        self.load_data()
        self._optimizer = optim.SGD(self.model.parameters(), lr=0.0005)
        pass

    def load_data(self):
        train_path = r'C:\workspace\2450\2450_clean_sub_delete.csv'
        test_path = r'C:\workspace\2720\2720_clean_sub_delete.csv'
        self.train_data_loader = DataLoader(train_path)
        self.test_data_loader = DataLoader(test_path)

    def train(self):
        self.model.train()
        for i in range(100):
            loss_sum = 0
            for input, target in zip(self.train_data_loader.train_input, self.train_data_loader.train_target):
                input = input.to(torch.float32)
                target = target.long()
                output = self.model(input)
                loss = self.model.loss(input, target)
                # Backward and optimize
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                loss_sum += loss
            print('epoch:{}    loss:{}'.format(i, loss_sum))
        torch.save(self.model.state_dict(), './model.pt')
        pass

    def predict(self):
        self.model.load_state_dict(torch.load('./model.pt'))
        sum = 0
        true = 0
        for input, target in zip(self.test_data_loader.train_input, self.test_data_loader.train_target):
            input = input.to(torch.float32)
            target = target.long()
            output = self.model(input)
            output = torch.argmax(output, dim=-1)
            x = target.eq(output)
            for i in range(len(x)):
                sum += 1
                if x[i]:
                    true += 1
        print('不同机器 准确率：{}'.format(true/sum))
        sum = 0
        true = 0
        for input, target in zip(self.train_data_loader.test_input, self.train_data_loader.test_target):
            input = input.to(torch.float32)
            target = target.long()
            output = self.model(input)
            output = torch.argmax(output, dim=-1)
            x = target.eq(output)
            for i in range(len(x)):
                sum += 1
                if x[i]:
                    true += 1
        print('相同机器 准确率：{}'.format(true / sum))
        pass


if __name__ == '__main__':
    ok_nok = OkNok()
    ok_nok.load_data()
    ok_nok.train()
    ok_nok.predict()
