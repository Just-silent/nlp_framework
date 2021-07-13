# -*- coding: utf-8 -*-
# @Author   : Just-silent
# @time     : 2020/9/18 8:49

import json
from torchtext.data import Dataset, Example

from sequence.emo.utils import Tool


class ODDataset(Dataset):

    def __init__(self, path, fields, file, config, **kwargs):
        self._config = config
        self._tool = Tool()
        examples = self._get_examples(path, fields, file)
        super(ODDataset, self).__init__(examples, fields, **kwargs)


    def _get_examples(self, path, fields, file):
        examples = []
        train_file = open(path, 'r', encoding='utf-8')
        for line in train_file.readlines():
            json_data = json.loads(line.strip())
            query1 = json_data['query-01']
            response1 = json_data['response-01']
            query2 = json_data['query-02']
            query_rewrite = ''
            if file != 'test':
                query_rewrite = json_data['query-02-rewrite']
            enc_text_list = ['[CLS]'] + [c for c in query1] + ['[SEP]'] + [c for c in response1] + ['[SEP]'] + [c for c in query2] + ['[SEP]']
            dec_text_list = ['[CLS]'] + [c for c in query_rewrite]
            tag_list = [c for c in query_rewrite] + ['[SEP]']
            examples.append(Example.fromlist([enc_text_list, dec_text_list, tag_list], fields))
        return examples