# -*- coding: utf-8 -*-
# @Author   : Just-silent
# @time     : 2020/9/18 13:55

import torch
from sklearn.metrics import classification_report, f1_score

from common.evaluation.common_sequence_evaluator import CommonSeqEvaluator


class EmotEvaluator(CommonSeqEvaluator):

    def __init__(self, config, tag_vocab):
        super(EmotEvaluator, self).__init__()
        self.tag_num = 0
        self.true_num = 0

    def evaluate(self, pred, target):
        # 送入batch数据
        self.tag_num += len(target)
        for a, b in zip(target, pred):
            if a == b:
                self.true_num += 1

    def get_eval_output(self):
        p = round(self.true_num/self.tag_num, 4)
        print('准确率：{}%'.format(p*100))
        return p
