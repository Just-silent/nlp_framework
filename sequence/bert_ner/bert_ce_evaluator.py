# -*- coding: utf-8 -*-
# @Author   : Just-silent
# @time     : 2020/9/18 13:55

import torch
from sklearn.metrics import classification_report, f1_score

from common.evaluation.common_sequence_evaluator import CommonSeqEvaluator


class EventExtractEvaluator(CommonSeqEvaluator):
    def __init__(self, config, idx2tag):
        super(EventExtractEvaluator, self).__init__()
        self.idx2tag = idx2tag
        self._average_type = config.evaluation.type
        self._labels = list(idx2tag.values())
        self._labels.remove('O')
        self._labels.remove('PAD')
        self._pred_list = []
        self._true_list = []

    def _change_type(self, preds, targets):
        pred_list = []
        target_list = []
        lens = []
        for target in targets:
            target1 = []
            for tag in target:
                if tag==0:
                    break
                else:
                    target1.append(tag)
            lens.append(len(target1))
            target_list.extend([self.idx2tag[index.item()] for index in target1])
        for i, pred in enumerate(preds):
            pred = pred[:lens[i]]
            pred_list.extend([self.idx2tag[index] for index in pred])
        return pred_list, target_list

    def evaluate(self, pred, target):
        # 送入batch数据
        pred, target = self._change_type(pred, target)
        self._pred_list.extend(pred)
        self._true_list.extend(target)

    def get_eval_output(self):
        result = classification_report(self._true_list, self._pred_list, labels=self._labels, digits=3, output_dict=False)
        f1 = f1_score(self._true_list, self._pred_list, labels=self._labels, average='micro')
        print(result)
        self._pred_list = []
        self._true_list = []
        return f1
