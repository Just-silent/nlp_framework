#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : metric
# @Author   : LiuYan
# @Time     : 2020/7/30 23:14

from __future__ import print_function
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_ner_category(golds, predicts, rights):
    gold_list = [gold.split(']')[-1] for gold in golds]
    pred_list = [pred.split(']')[-1] for pred in predicts]
    right_list = [right.split(']')[-1] for right in rights]
    arr_set = set(gold_list)
    gold_dict, pred_dict, right_dict = {'total': len(gold_list)}, {'total': len(pred_list)}, {'total': len(right_list)}
    for arr in arr_set:
        gold_dict.update({arr: gold_list.count(arr)})
        pred_dict.update({arr: pred_list.count(arr)})
        right_dict.update({arr: right_list.count(arr)})
    print('\n' + ''.join('-' for _ in range(89)))
    print('label_type\t\t\tp\t\t\tr\t\t\tf1\t\t\tright_num\tpred_num\tlabel_num')
    strs = '{0:}{1:<12.4f}{2:<12.4f}{3:<12.4f}{4:<12}{5:<12}{6:<10}'
    for arr in arr_set:
        p = right_dict[arr] / pred_dict[arr] if pred_dict[arr] != 0 else 0
        r = right_dict[arr] / gold_dict[arr] if gold_dict[arr] != 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) != 0 else 0
        arr_out = arr + ''.join(
            ' ' for i in range(20 - (((len(arr.encode('utf-8')) - len(arr)) // 2) + len(arr))))
        print(strs.format(
            arr_out, p, r, f1, right_dict[arr], pred_dict[arr], gold_dict[arr]
        ), chr(12288))
    p = right_dict['total'] / pred_dict['total'] if pred_dict['total'] != 0 else 0
    r = right_dict['total'] / gold_dict['total'] if gold_dict['total'] != 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) != 0 else 0
    print(strs.format(
        'average{}'.format(''.join(' ' for i in range(13))),
        p, r, f1, right_dict['total'], pred_dict['total'], gold_dict['total']
    ), chr(12288))
    print(''.join('-' for _ in range(89)) + '\n')
    pass


# input as sentence level labels
def get_ner_fmeasure(golden_lists, predict_lists, label_type='BMES', plot_cm=False, type_list=None):
    sent_num = len(golden_lists)
    golden_full, predict_full, right_full = [], [], []
    true_list_all, pred_list_all = [], []
    right_tag = 0
    all_tag = 0
    for idx in range(0, sent_num):
        golden_list = golden_lists[idx]
        predict_list = predict_lists[idx]
        for idy in range(len(golden_list)):
            if golden_list[idy] == predict_list[idy]:
                right_tag += 1
        all_tag += len(golden_list)
        if label_type == 'BMES' or label_type == 'BIOES':
            gold_matrix = get_ner_BMES(golden_list)
            pred_matrix = get_ner_BMES(predict_list)
        else:
            gold_matrix = get_ner_BIO(golden_list)
            pred_matrix = get_ner_BIO(predict_list)
        right_ner = list(set(gold_matrix).intersection(set(pred_matrix)))
        golden_full += gold_matrix
        predict_full += pred_matrix
        right_full += right_ner
        true_list_all.append(gold_matrix)
        pred_list_all.append(pred_matrix)
    right_num = len(right_full)
    golden_num = len(golden_full)
    predict_num = len(predict_full)
    if predict_num == 0:
        precision = -1
    else:
        precision = (right_num + 0.0) / predict_num
    if golden_num == 0:
        recall = -1
    else:
        recall = (right_num + 0.0) / golden_num
    if (precision == -1) or (recall == -1) or (precision + recall) <= 0.:
        f_measure = -1
    else:
        f_measure = 2 * precision * recall / (precision + recall)
    accuracy = (right_tag + 0.0) / all_tag
    # print 'Accuracy: ', right_tag,'/',all_tag,'=',accuracy
    # if label_type.upper().startswith("B-"):
    #     print('gold_num = ', golden_num, ' pred_num = ', predict_num, ' right_num = ', right_num)
    # else:
    #     print('Right token = ', right_tag, ' All token = ', all_tag, ' acc = ', accuracy)

    if label_type in ['BMES', 'BIOES', 'BIO']:
        get_ner_category(golds=golden_full, predicts=predict_full, rights=right_full)
    if plot_cm:
        confusion_matrix = stats_confusion_matrix(
            true_list_all=true_list_all, pred_list_all=pred_list_all, type_list=type_list
        )
        confusion_matrix = select_confusion_matrix(confusion_matrix=confusion_matrix)
        plot_confusion_matrix(confusion_matrix=confusion_matrix)

    return accuracy, precision, recall, f_measure


def reverse_style(input_string):
    target_position = input_string.index('[')
    input_len = len(input_string)
    output_string = input_string[target_position:input_len] + input_string[0:target_position]
    return output_string


def stats_confusion_matrix(true_list_all, pred_list_all, type_list):
    assert type(true_list_all) is list, 'type(true_list_all) is not list but {}'.format(type(true_list_all))
    assert type(pred_list_all) is list, 'type(pred_list_all) is not list but {}'.format(type(pred_list_all))
    assert len(true_list_all) == len(pred_list_all), 'true_list_all: {} != pred_list_all: {} !'.format(
        len(true_list_all), len(pred_list_all)
    )
    pred_list = copy.deepcopy(type_list)
    pred_list.append('UNK')
    confusion_matrix = dict()
    for chunk_type in type_list:
        confusion_matrix[chunk_type] = dict().fromkeys(pred_list, 0)

    # compute confusion_matrix
    for true_list, pred_list in zip(true_list_all, pred_list_all):
        for true in true_list:
            dis_bool = False
            true_index, true_type = true.split(']')
            for pred in pred_list:
                pred_index, pred_type = pred.split(']')
                if true_index == pred_index:
                    dis_bool = True
                    if true_type == pred_type:
                        confusion_matrix[true_type][true_type] += 1
                    else:
                        confusion_matrix[true_type][pred_type] += 1
                    break
            if not dis_bool:
                confusion_matrix[true_type]['UNK'] += 1

    return confusion_matrix
    pass


def plot_confusion_matrix(confusion_matrix):
    true_key_list = list(confusion_matrix.keys())
    pred_key_list = list(confusion_matrix.keys())
    pred_key_list.append('Other')
    pred_key_list.append('UNK')
    matrix_list = list()
    for type in true_key_list:
        type_list = list(confusion_matrix[type].values())
        matrix_list.append(type_list)
    plot(matrix_list, true_key_list, pred_key_list)
    pass


def plot(matrix_list, true_key_list: list, pred_key_list: list):
    matrix = np.array(matrix_list)
    # print(matrix)
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    con_mat_norm = np.around(matrix, decimals=2)
    sns.set()
    f, ax = plt.subplots()
    matrix = pd.DataFrame(con_mat_norm, index=true_key_list, columns=pred_key_list)
    # print(matrix)
    sns.heatmap(matrix, annot=True, cmap='Blues', ax=ax)  # draw heat map
    ax.set_title('confusion matrix')  # title
    ax.set_xlabel('predict')  # x-axis
    ax.set_ylabel('true')  # y-axis
    plt.show()
    pass


def select_confusion_matrix(confusion_matrix):
    con_mat = dict()
    select_dict = {
        'SIMPLE_CHEMICAL': 'CHEM',
        'CELLULAR_COMPONENT': 'CC',
        'GENE_OR_GENE_PRODUCT': 'GGP',
        'CELL': 'CELL',
        'ORGANISM': 'SPE',
    }
    select_list = list(select_dict.keys())
    for select_type in select_list:
        cm_dict = confusion_matrix[select_type]
        cm_list = list(cm_dict.keys())
        cm_dict_ = {
            'CHEM': int(), 'CC': int(), 'GGP': int(),
            'CELL': int(), 'SPE': int()
        }
        other = 0
        for type_ in cm_list:
            if type_ in select_list:
                cm_dict_[select_dict[type_]] = cm_dict[type_]
            else:
                other += cm_dict[type_]
        cm_dict_['Other'] = other - cm_dict['UNK']
        cm_dict_['UNK'] = cm_dict['UNK']
        con_mat[select_dict[select_type]] = cm_dict_
    return con_mat
    pass


def get_ner_BMES(label_list):
    list_len = len(label_list)
    begin_label = 'B-'
    end_label = 'E-'
    single_label = 'S-'
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []
    for i in range(0, list_len):
        current_label = label_list[i].upper()
        if begin_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i - 1))
            whole_tag = current_label.replace(begin_label, '', 1) + '[' + str(i)
            index_tag = current_label.replace(begin_label, '', 1)

        elif single_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i - 1))
            whole_tag = current_label.replace(single_label, '', 1) + '[' + str(i)
            tag_list.append(whole_tag)
            whole_tag = ''
            index_tag = ''
        elif end_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i))
            whole_tag = ''
            index_tag = ''
        else:
            continue
    if (whole_tag != '') & (index_tag != ''):
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i] + ']'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)
    return stand_matrix


def get_ner_BIO(label_list):
    list_len = len(label_list)
    begin_label = 'B-'
    inside_label = 'I-'
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []
    for i in range(0, list_len):
        current_label = label_list[i].upper()
        if begin_label in current_label:
            if index_tag == '':
                whole_tag = current_label.replace(begin_label, "", 1) + '[' + str(i)
                index_tag = current_label.replace(begin_label, "", 1)
            else:
                tag_list.append(whole_tag + ',' + str(i - 1))
                whole_tag = current_label.replace(begin_label, "", 1) + '[' + str(i)
                index_tag = current_label.replace(begin_label, "", 1)

        elif inside_label in current_label:
            if current_label.replace(inside_label, "", 1) == index_tag:
                whole_tag = whole_tag
            else:
                if (whole_tag != '') & (index_tag != ''):
                    tag_list.append(whole_tag + ',' + str(i - 1))
                whole_tag = ''
                index_tag = ''
        else:
            if (whole_tag != '') & (index_tag != ''):
                tag_list.append(whole_tag + ',' + str(i - 1))
            whole_tag = ''
            index_tag = ''

    if (whole_tag != '') & (index_tag != ''):
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i] + ']'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)
    return stand_matrix


def readSentence(input_file):
    in_lines = open(input_file, 'r').readlines()
    sentences = []
    labels = []
    sentence = []
    label = []
    for line in in_lines:
        if len(line) < 2:
            sentences.append(sentence)
            labels.append(label)
            sentence = []
            label = []
        else:
            pair = line.strip('\n').split(' ')
            sentence.append(pair[0])
            label.append(pair[-1])
    return sentences, labels


def readTwoLabelSentence(input_file, pred_col=-1):
    in_lines = open(input_file, 'r').readlines()
    sentences = []
    predict_labels = []
    golden_labels = []
    sentence = []
    predict_label = []
    golden_label = []
    for line in in_lines:
        if '##score##' in line:
            continue
        if len(line) < 2:
            sentences.append(sentence)
            golden_labels.append(golden_label)
            predict_labels.append(predict_label)
            sentence = []
            golden_label = []
            predict_label = []
        else:
            pair = line.strip('\n').split(' ')
            sentence.append(pair[0])
            golden_label.append(pair[1])
            predict_label.append(pair[pred_col])

    return sentences, golden_labels, predict_labels


def fmeasure_from_file(golden_file, predict_file, label_type='BMES'):
    print('Get f measure from file:', golden_file, predict_file)
    print('Label format:', label_type)
    golden_sent, golden_labels = readSentence(golden_file)
    predict_sent, predict_labels = readSentence(predict_file)
    P, R, F = get_ner_fmeasure(golden_labels, predict_labels, label_type)
    print('P:%sm R:%s, F:%s' % (P, R, F))


def fmeasure_from_singlefile(twolabel_file, label_type='BMES', pred_col=-1):
    sent, golden_labels, predict_labels = readTwoLabelSentence(twolabel_file, pred_col)
    P, R, F = get_ner_fmeasure(golden_labels, predict_labels, label_type)
    print('P:%s, R:%s, F:%s' % (P, R, F))


if __name__ == '__main__':
    right_full = ['[1]PER', '[2,3]PER', '[1]ORG', '[3]LOC']
    gold_full = ['[1]PER', '[2,3]PER', '[1]ORG', '[3]LOC', '[5]ORG']
    pred_full = ['[1]PER', '[2,3]PER', '[1]ORG', '[3]LOC', '[2]ORG']
    get_ner_category(gold_full, pred_full, right_full)
    pass
