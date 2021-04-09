# -*- coding: utf-8 -*-
# @Author   : Just-silent
# @time     : 2020/9/18 8:49

import json
from tqdm import tqdm
from openpyxl import load_workbook
from torchtext.data import Dataset, Example


class EEDataset(Dataset):

    def __init__(self, path, fields, file, tree, config, encoding="utf-8", **kwargs):
        self._config = config
        examples = self._get_examples(path, fields, file)
        super(EEDataset, self).__init__(examples, fields, **kwargs)

    def _get_examples(self, path, fields, file):
        examples = []
        examples.append(Example.fromlist([sentence_list, tag_list], fields))
        return examples