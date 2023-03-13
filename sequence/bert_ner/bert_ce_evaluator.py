# -*- coding: utf-8 -*-
# @Author   : Just-silent
# @time     : 2020/9/18 13:55

import torch
from sklearn.metrics import classification_report, f1_score

from common.evaluation.common_sequence_evaluator import CommonSeqEvaluator


# class EventExtractEvaluator(CommonSeqEvaluator):
#     def __init__(self, config, idx2tag):
#         super(EventExtractEvaluator, self).__init__()
#         self.idx2tag = idx2tag
#         self._average_type = config.evaluation.type
#         self._labels = list(idx2tag.values())
#         self._labels.remove('O')
#         self._labels.remove('PAD')
#         self._pred_list = []
#         self._true_list = []
#
#     def _change_type(self, preds, targets):
#         pred_list = []
#         target_list = []
#         lens = []
#         for target in targets:
#             target1 = []
#             for tag in target:
#                 if tag==0:
#                     break
#                 else:
#                     target1.append(tag)
#             lens.append(len(target1))
#             target_list.extend([self.idx2tag[index.item()] for index in target1])
#         for i, pred in enumerate(preds):
#             pred = pred[:lens[i]]
#             pred_list.extend([self.idx2tag[index] for index in pred])
#         return pred_list, target_list
#
#     def evaluate(self, pred, target):
#         # 送入batch数据
#         pred, target = self._change_type(pred, target)
#         self._pred_list.extend(pred)
#         self._true_list.extend(target)
#
#     def get_eval_output(self):
#         result = classification_report(self._true_list, self._pred_list, labels=self._labels, digits=3, output_dict=False)
#         f1 = f1_score(self._true_list, self._pred_list, labels=self._labels, average='micro')
#         print(result)
#         self._pred_list = []
#         self._true_list = []
#         return f1

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
        self.tags = [("B-defect", "I-defect")]

    def precision(self, pre_labels, true_labels):
        '''
        :param pre_tags: list
        :param true_tags: list
        :return:
        '''
        pre = []
        pre_result = self.find_all_tag(pre_labels)  # pre_result是一个字典，键是标签，值是一个元组，第一位是B的位置，第二位是长度
        for name in pre_result:  # 取得键，也就是标签
            for x in pre_result[name]:  # 取得值：也就是元组，注意元组可能有多个
                if x:  # 如果x存在
                    if pre_labels[x[0]:x[0] + x[1]] == true_labels[x[0]:x[0] + x[1]]:  # 判断对应位置的每个标签是否一致
                        pre.append(1)  # 一致则结果添加1
                    else:
                        pre.append(0)  # 不一致则结果添加0
        return sum(pre) / len(pre)  # 为1的个数/总个数

    def recall(self, pre_labels, true_labels):
        '''
        :param pre_tags: list
        :param true_tags: list
        :return:
        '''
        recall = []
        if isinstance(pre_labels, str):
            pre_labels = pre_labels.strip().split()
            pre_labels = ["O" if label == "0" else label for label in pre_labels]
        if isinstance(true_labels, str):
            true_labels = true_labels.strip().split()
            true_labels = ["O" if label == "0" else label for label in true_labels]

        true_result = self.find_all_tag(true_labels)
        for name in true_result:  # 取得键，也就是标签，这里注意和计算precision的区别，遍历的是真实标签列表
            for x in true_result[name]:  # 以下的基本差不多
                if x:
                    if pre_labels[x[0]:x[0] + x[1]] == true_labels[x[0]:x[0] + x[1]]:
                        recall.append(1)
                    else:
                        recall.append(0)
        return sum(recall) / len(recall)

    def f1_score(self, precision, recall):
        return (2 * precision * recall) / (precision + recall)  # 有了precision和recall，计算F1就简单了

    def find_tag(self, labels, B_label, I_label):
        result = []
        for num in range(len(labels)):  # 遍历Labels
            if labels[num] == B_label:
                song_pos0 = num  # 记录B_SONG的位置
            if labels[num] == I_label and labels[num - 1] == B_label:  # 如果当前lable是I_SONG且前一个是B_SONG
                lenth = 2  # 当前长度为2
                for num2 in range(num, len(labels)):  # 从该位置开始继续遍历
                    if labels[num2] == I_label and labels[num2 - 1] == I_label:  # 如果当前位置和前一个位置是I_SONG
                        lenth += 1  # 长度+1
                    if labels[num2] == "O":  # 如果当前标签是O
                        result.append((song_pos0, lenth))  # z则取得B的位置和长度
                        break  # 退出第二个循环
        return result

    def find_all_tag(self, labels):
        result = {}
        for tag in self.tags:
            res = self.find_tag(labels, B_label=tag[0], I_label=tag[1])
            result[tag[0].split("-")[1]] = res  # 将result赋值给就标签
        return result

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
        precision = self.precision(self._pred_list, self._true_list)
        recall = self.recall(self._pred_list, self._true_list)
        f1 = self.f1_score(precision, recall)
        print('准确率：', precision)
        print('召回率', recall)
        print('F1', f1)
        self._pred_list = []
        self._true_list = []
        return f1