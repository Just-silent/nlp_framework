# -*- coding: utf-8 -*-
# @Author   : Just-silent
# @time     : 2020/9/18 8:49
import json

from openpyxl import load_workbook
from torchtext.data import Dataset, Example

from sequence.bert_ner.utils import Tool


class EEDataset(Dataset):
    def __init__(self, path, fields, file, config, **kwargs):
        self._config = config
        self._tool = Tool()
        examples = self._get_examples(path, fields, file)
        super(EEDataset, self).__init__(examples, fields, **kwargs)

    def _get_examples(self, path, fields, file):
        examples = []
        datas = []
        with open(path, 'r', encoding='utf-8') as f:
            datas = f.readlines()
        for data in datas:
            data = json.load(data)
            # text = ws.cell(index_row, text_col).value
            # text_list = [c for c in text]
            # tag_list = self._tool.get_tags(text, words)
            examples.append(Example.fromlist([text_list, tag_list], fields))
        return examples