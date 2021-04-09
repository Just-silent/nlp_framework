#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : build_word2vec_weights
# @Author   : LiuYan
# @Time     : 2020/6/24 14:46

from itertools import islice

import numpy as np
import torch
from common.util.utils import timeit


@timeit
def load_word2vec(path=None, word_vocab=None, embedding_dim=None):
    """
    loading word vector
    :param path: None
    :param word_vocab: None
    :param embedding_dim: 768/100 bert/glove.6B.100d
    :return: a vector corresponding to word_vocab.
    """
    word_vocab_dict = word_vocab.stoi
    vectors_vocab = load_vec(path, embedding_dim=embedding_dim)
    if '[PAD]' in vectors_vocab:
        pad = vectors_vocab['[PAD]']
    elif 'pad' in vectors_vocab:
        pad = vectors_vocab['pad']
    if '[UNK]' in vectors_vocab:
        unk = vectors_vocab['[UNK]']
    elif 'unk' in vectors_vocab:
        unk = vectors_vocab['unk']
    vocab_size = len(word_vocab)
    embed_weights = torch.zeros(vocab_size, embedding_dim)
    for word, index in word_vocab_dict.items():  # word and index
        if word in vectors_vocab:
            em = vectors_vocab[word]
        elif word == '<pad>':
            em = pad
        else:
            em = unk
        embed_weights[index, :] = torch.from_numpy(np.array(em))
    return embed_weights


@timeit
def load_vec(path=None, embedding_dim=None):
    """
    loading word vector
    :param path: None
    :param embedding_dim: 768/100 bert/glove.6B.100d
    :return: a dictionary of word vectors
    """
    vectors_vocab = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in islice(f, 1, None):  # skip the first row
            items = line.split()
            char, vectors = items[0], items[-embedding_dim:]
            vectors = [float(vector) for vector in vectors]
            vectors_vocab[char] = vectors
    return vectors_vocab
