# -*- coding: utf-8 -*-
# @Author   : Just-silent
# @time     : 2020/9/18 8:16

import torch
import random
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer


class BertDataLoader(object):
    def __init__(self, config):
        # , data_dir, bert_class, params, token_pad_idx=0, tag_pad_idx=-1
        self._config = config
        self.batch_size = config.data.batch_size
        self.max_len = config.data.max_len
        self.token_pad_idx = config.data.token_pad_idx
        self.tag_pad_idx = config.data.tag_pad_idx
        self._shuffle = config.data.shuffle
        self.device = config.device
        self.seed = config.project.seed

        tags = self.load_tags()
        self.tag2idx = {tag: idx for idx, tag in enumerate(tags)}
        self.idx2tag = {idx: tag for idx, tag in enumerate(tags)}

        self.tokenizer = BertTokenizer.from_pretrained(config.pretrained_models.dir, do_lower_case=False)

        self.train_data = {}
        self.valid_data = {}
        self.test_data = {}

        self.load_data(config.data.train_path, self.train_data)
        self.load_data(config.data.valid_path, self.valid_data)
        self.load_data(config.data.test_path, self.test_data)

    def load_tags(self):
        tags = []
        with open(self._config.data.train_path, 'r', encoding='utf-8') as file:
            for line in file.readlines():
                if line!='\n':
                    tags.append(line.strip().split()[1])
        result = []
        if self._config.model.label_pad:
            result = ['PAD']
        tags_new = list(set(tags))
        tags_new.sort(key=tags.index)
        result.extend(tags_new)
        return result
        # return list(set(tags))

    def load_sentences_tags(self, path, data):
        """
        Args:
            path : the path of the data_file
            data : the list of data
        Returns:
            the list of all vocabs
        """
        sentences = []
        tags = []

        file = open(path, 'r', encoding='utf-8')
        sentence = []
        ts = []
        for line in tqdm(file.readlines()):
            if line=='\n':
                if sentence == []:
                    pass
                else:
                    subwords = list(map(self.tokenizer.tokenize, sentence))
                    subword_lengths = list(map(len, subwords))
                    subwords = ['[CLS]'] + [item for indices in subwords for item in indices]
                    token_start_idxs = 1 + np.cumsum([0] + subword_lengths[:-1])
                    sentences.append((self.tokenizer.convert_tokens_to_ids(subwords), token_start_idxs))
                    ts = [self.tag2idx.get(tag) for tag in ts]
                    tags.append(ts)
                    sentence = []
                    ts = []
            else:
                char, t = line.strip().split()
                sentence.append(char)
                ts.append(t)
        data['tags'] = tags
        data['data'] = sentences
        data['size'] = len(sentences)

    def load_data(self, path , data):
        """Loads the data for each type in types from data_dir.

        Args:
            data_type: (str) has one of 'train', 'val', 'test' depending on which data is required.
        Returns:
            data: (dict) contains the data with tags for each type in types.
        """
        self.load_sentences_tags(path, data)

    def data_iterator(self, data):
        """Returns a generator that yields batches data with tags.

        Args:
            data: (dict) contains data which has keys 'data', 'tags' and 'size'
            shuffle: (bool) whether the data should be shuffled

        Yields:
            batch_data: (tensor) shape: (batch_size, max_len)
            batch_tags: (tensor) shape: (batch_size, max_len)
        """

        # make a list that decides the order in which we go over the data- this avoids explicit shuffling of data
        order = list(range(data['size']))
        if self._shuffle:
            random.seed(self.seed)
            random.shuffle(order)

        interMode = False if 'tags' in data else True

        if data['size'] % self.batch_size == 0:
            BATCH_NUM = data['size'] // self.batch_size
        else:
            BATCH_NUM = data['size'] // self.batch_size + 1

        # one pass over data
        for i in range(BATCH_NUM):
            # fetch sentences and tags
            if i * self.batch_size < data['size'] < (i + 1) * self.batch_size:
                sentences = [data['data'][idx] for idx in order[i * self.batch_size:]]
                if not interMode:
                    tags = [data['tags'][idx] for idx in order[i * self.batch_size:]]
            else:
                sentences = [data['data'][idx] for idx in order[i * self.batch_size:(i + 1) * self.batch_size]]
                if not interMode:
                    tags = [data['tags'][idx] for idx in order[i * self.batch_size:(i + 1) * self.batch_size]]

            # batch length
            batch_len = len(sentences)

            # compute length of longest sentence in batch
            batch_max_subwords_len = max([len(s[0]) for s in sentences])
            max_subwords_len = min(batch_max_subwords_len, self.max_len)
            max_token_len = 0

            # prepare a numpy array with the data, initialising the data with pad_idx
            batch_data = self.token_pad_idx * np.ones((batch_len, max_subwords_len))
            batch_token_starts = []

            # copy the data to the numpy array
            for j in range(batch_len):
                cur_subwords_len = len(sentences[j][0])
                if cur_subwords_len <= max_subwords_len:
                    batch_data[j][:cur_subwords_len] = sentences[j][0]
                else:
                    batch_data[j] = sentences[j][0][:max_subwords_len]
                token_start_idx = sentences[j][-1]
                token_starts = np.zeros(max_subwords_len)
                token_starts[[idx for idx in token_start_idx if idx < max_subwords_len]] = 1
                batch_token_starts.append(token_starts)
                max_token_len = max(int(sum(token_starts)), max_token_len)

            if not interMode:
                batch_tags = self.tag_pad_idx * np.ones((batch_len, max_token_len))
                for j in range(batch_len):
                    cur_tags_len = len(tags[j])
                    if cur_tags_len <= max_token_len:
                        batch_tags[j][:cur_tags_len] = tags[j]
                    else:
                        batch_tags[j] = tags[j][:max_token_len]

            # since all data are indices, we convert them to torch LongTensors
            batch_data = torch.tensor(batch_data, dtype=torch.long)
            batch_token_starts = torch.tensor(batch_token_starts, dtype=torch.long)
            if not interMode:
                batch_tags = torch.tensor(batch_tags, dtype=torch.long)

            # shift tensors to GPU if available
            batch_data, batch_token_starts = batch_data.to(self.device), batch_token_starts.to(self.device)
            if not interMode:
                batch_tags = batch_tags.to(self.device)
                yield batch_data, batch_token_starts, batch_tags
            else:
                yield batch_data, batch_token_starts

    def load_train(self):
        return self.train_data

    def load_valid(self):
        return self.valid_data

    def load_test(self):
        return self.test_data


if __name__ == '__main__':
    config_file = 'bert_ce_config.yml'
    import dynamic_yaml

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open(config_file, mode='r', encoding='UTF-8') as f:
        config = dynamic_yaml.load(f)
    data_loader = BertDataLoader(config)
    datas = data_loader.load_data()
    pass