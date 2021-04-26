# -*- coding: utf-8 -*-
# @Author   : Just-silent
# @time     : 2020/9/18 13:55

import torch
from sklearn.metrics import classification_report, f1_score

from common.evaluation.common_sequence_evaluator import CommonSeqEvaluator


class TextSimilarityEvaluator(CommonSeqEvaluator):
    def __init__(self, config):
        super(TextSimilarityEvaluator, self).__init__()
        self._config = config
        # self._labels.remove('O')
        # self._labels.remove('PAD')
        self._pred_list = []
        self._true_list = []

    def _change_type(self, preds, targets):
        if self._config.device=='cpu':
            preds = preds.numpy().tolist()
            targets = targets.numpy().tolist()
        else:
            preds = preds.cpu().numpy().tolist()
            targets = targets.cpu().numpy().tolist()
        return preds, targets

    def evaluate(self, pred, target):
        # 送入batch数据
        pred, target = self._change_type(pred, target)
        self._pred_list.extend(pred)
        self._true_list.extend(target)

    def get_eval_output(self):
        result = classification_report(self._true_list, self._pred_list, labels=[0, 1], digits=3,
                                       output_dict=False)
        f1 = f1_score(self._true_list, self._pred_list, labels=[0, 1], average='micro')
        print(result)
        self._pred_list = []
        self._true_list = []
        return f1
