#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : common_loss
# @Author   : Xiaoming Liu
# @Time     : 2020/09/15 15:55
from abc import ABC, abstractmethod

from base.loss.base_loss import BaseLoss


class CommonLoss(BaseLoss, ABC):

    def __init__(self, loss_config):
        super(CommonLoss, self).__init__(loss_config)
        self._config = loss_config
        pass

    @abstractmethod
    def forward(self, dict_outputs: dict) -> dict:
        pass


class CELoss():
    def __init__(self):
        pass

    def forward(self):
        pass


class DiceLoss():
    def __init__(self):
        pass

    def forward(self):
        pass