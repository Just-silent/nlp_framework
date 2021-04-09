# -*- coding: utf-8 -*-
# @Author   : Just-silent
# @time     : 2020/9/16 16:21


from abc import ABC, abstractmethod
from base.evaluation.base_sequence_evaluator import BaseSeqEvaluator


class CommonSeqEvaluator(BaseSeqEvaluator):

    def __init__(self):
        super(CommonSeqEvaluator, self).__init__()

    def _change_type(self, pred, target):
        # change the type of input
        pass

    @abstractmethod
    def evaluate(self, pred, target):
        # send data to this by batch
        pass

    def _get_eval_result(self):
        # Count the eval results from all batch data
        pass

    @abstractmethod
    def get_eval_output(self):
        # 外部获取结果接口,并且可以配置是否打印（eval结果保存暂时默认保存）
        pass

    def _print_table(self, List : list):
        # display result
        pass

    def _write_csv(self, List : list):
        # write the result to csv
        pass