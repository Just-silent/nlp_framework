# -*- coding: utf-8 -*-
# @Author   : Just-silent
# @time     : 2020/9/18 8:16

import torch
from torchtext.data import Field, BucketIterator, LabelField
from torchtext.vocab import Vectors, Vocab
from transformers import BertConfig, BertTokenizer, BertModel

from common.util.utils import timeit
from sequence.emo.utils import Tool
from sequence.emo.emo_dataset import EmoDataset
from common.data.common_data_loader import CommonDataLoader


def tokenizer(token):
    return [k for k in token]


START_TAG = "<START>"
STOP_TAG = "<STOP>"
PAD_TAG = "<PAD>"
UNK_TAG = "<PAD>"


class SequenceDataLoader(CommonDataLoader):

    def __init__(self, data_config):
        super(SequenceDataLoader, self).__init__(data_config)
        self._config = data_config
        self._tool = Tool()
        self.__build_field()
        self._load_data()
        pass

    def __build_field(self):
        self.TEXT = Field(sequential=True, use_vocab=True, lower=True, tokenize=tokenizer, include_lengths=True,
                          batch_first=self._config.data.batch_first, pad_token='[PAD]', unk_token='[UNK]')
        # self.TAG = Field(sequential=True, use_vocab=True, tokenize=tokenizer, is_target=True,
        #                  batch_first=self._config.data.batch_first)
        self.TAG = LabelField(sequential=True, use_vocab=True, tokenize=tokenizer, is_target=True,)
        self._fields = [
            ('text', self.TEXT), ('tag', self.TAG)
        ]
        pass

    @timeit
    def _load_data(self):
        self.train_data = EmoDataset(
            path=self._config.data.train_path, fields=self._fields,
            file='train', config=self._config
        )
        self.valid_data = EmoDataset(
            path=self._config.data.valid_path, fields=self._fields,
            file='valid', config=self._config
        )
        self.test_data = EmoDataset(
            path=self._config.data.test_path, fields=self._fields,
            file='test', config=self._config
        )
        self.__build_vocab(self.train_data, self.valid_data, self.test_data)
        self.__build_iterator(self.train_data, self.valid_data, self.test_data)
        pass

    def __build_vocab(self, *dataset):
        """
        :param dataset: train_data, valid_data, test_data
        :return: text_vocab, tag_vocab
        """
        if self._config.pretrained_models.is_use:
            vocabs = self._tool.get_vocab_list(self._config.data.vocab_path)
            v = Vocab(vocabs, specials=['[PAD]', '[UNK]'])
            self.TEXT.build_vocab(vocabs,
                                  max_size=30000,
                                  min_freq=1,
                                  vectors=None,  # vects替换为None则不使用词向量
                                  )
        else:
            self.TEXT.build_vocab(*dataset)
        self.TAG.build_vocab(*dataset)
        self.word_vocab = self.TEXT.vocab
        self.tag_vocab = self.TAG.vocab
        pass

    def __build_iterator(self, *dataset):
        self._train_iter = BucketIterator(
            dataset[0], batch_size=self._config.data.train_batch_size, shuffle=True,
            sort_key=lambda x: len(x.text), sort_within_batch=True, device=self._config.device)
        self._valid_iter = BucketIterator(
            dataset[1], batch_size=self._config.data.train_batch_size, shuffle=False,
            sort_key=lambda x: len(x.text), sort_within_batch=True, device=self._config.device)
        self._test_iter = BucketIterator(
            dataset[2], batch_size=self._config.data.train_batch_size, shuffle=False,
            sort_key=lambda x: len(x.text), sort_within_batch=True, device=self._config.device)
        pass

    def load_train(self):
        return self._train_iter
        pass

    def load_test(self):
        return self._test_iter
        pass

    def load_valid(self):
        return self._valid_iter
        pass


if __name__ == '__main__':
    config_file = 'emo_config.yml'
    import dynamic_yaml

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open(config_file, mode='r', encoding='UTF-8') as f:
        config = dynamic_yaml.load(f)
    data_loader = SequenceDataLoader(config)
    for batch, batch_data in enumerate(data_loader.load_train(), 0):
        text = batch_data.text
        print("batch = {}".format(batch))
        for idx, txt in enumerate(text, 0):
            print("idx={},text ={} ".format(idx, txt))

    pass
