# -*- coding: utf-8 -*-
# @Author   : Just-silent
# @time     : 2020/9/18 13:55

import torch
from sklearn.metrics import classification_report, f1_score

from common.evaluation.common_sequence_evaluator import CommonSeqEvaluator


class IntentionClassificationEvaluator(CommonSeqEvaluator):
    def __init__(self, config, idx2tag):
        super(IntentionClassificationEvaluator, self).__init__()
        self.idx2tag = idx2tag
        self._average_type = config.evaluation.type
        self._labels = list(idx2tag.values())
        # self._labels.remove('O')
        # self._labels.remove('PAD')
        self._pred_list = []
        self._true_list = []

    def _change_type(self, preds, targets):
        return preds, targets

    def evaluate(self, pred, target):
        # 送入batch数据
        pred, target = self._change_type(pred, target)
        self._pred_list.extend(pred)
        self._true_list.extend(target)

    def get_eval_output(self):
        nums = len(self._pred_list)
        true_nums = 0
        for i in range(nums):
            if self._true_list[i]==self._pred_list[i]:
                true_nums+=1
        p = true_nums/nums
        print('准确率：{}'.format(p))
        self._pred_list = []
        self._true_list = []
        return p
