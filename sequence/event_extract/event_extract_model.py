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

from common.model.common_model import CommonModel

RANDOM_SEED = 2020

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


class EventExteactModel(CommonModel):

    def __init__(self, seq_config_file):
        self._config = None
        super(EventExteactModel, self).__init__(seq_config_file)


    def forward(self, dict_inputs: dict) -> dict:
        pass