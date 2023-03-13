# -*- coding:utf-8 _*-
"""
@version: 
author: just_silent
@time: 2022/08/10
@file: tool.py
@function: 
@modify: 
"""
import json
import random

import pandas as pd
from tqdm import tqdm

# json->conll

# path = r'C:\workspace\data\qirui\data.json'
# train_path = r'C:\workspace\data\qirui\train_data.json'
# dev_path = r'C:\workspace\data\qirui\dev_data.json'
# datas = []
# with open(path, 'r', encoding='utf-8') as f:
#     datas = f.readlines()
# data_indexs = list(range(0, len(datas)))
# random.shuffle(data_indexs)
# train_f = open(train_path, 'w', encoding='utf-8')
# dev_f = open(dev_path, 'w', encoding='utf-8')
# mid_index = int(len(datas)*8/10)
# new_train_datas = []
# new_dev_datas = []
# for i in range(0, mid_index):
#     new_train_datas.append(datas[data_indexs[i]])
# for i in range(mid_index, len(datas)):
#     new_dev_datas.append(datas[data_indexs[i]])
# train_f.writelines(new_train_datas)
# train_f.close()
# dev_f.writelines(new_dev_datas)
# dev_f.close()

# train_save_path = r'C:\workspace\data\qirui\train_data.txt'
# dev_save_path = r'C:\workspace\data\qirui\dev_data.txt'
# train_datas = []
# dev_datas = []
#
# datas = []
# with open(train_path, 'r', encoding='utf-8') as f:
#     datas = f.readlines()
# for data in tqdm(datas):
#     data_dict = json.loads(data)
#     text = data_dict['text']
#     start_index = data_dict['h']['pos'][0]
#     end_index = data_dict['h']['pos'][1]
#     target = text[start_index:end_index]
#     i = 0
#     while i < len(text):
#         if text[i] == target[0] and i + len(target) < len(text) and text[i:i + len(target)]:
#             train_datas.append('{}\t{}\n'.format(text[i], 'B'))
#             num = i
#             for i in range(num + 1, num + len(target)):
#                 train_datas.append('{}\t{}\n'.format(text[i], 'I'))
#         else:
#             train_datas.append('{}\t{}\n'.format(text[i], 'O'))
#         i += 1
#         if i >= len(text):
#             i = 0
#             break
#     train_datas.append('\n')
#
# datas = []
# with open(dev_path, 'r', encoding='utf-8') as f:
#     datas = f.readlines()
# for data in tqdm(datas):
#     data_dict = json.loads(data)
#     text = data_dict['text']
#     start_index = data_dict['h']['pos'][0]
#     end_index = data_dict['h']['pos'][1]
#     target = text[start_index:end_index]
#     i = 0
#     while i < len(text):
#         if text[i] == target[0] and i + len(target) < len(text) and text[i:i + len(target)]:
#             train_datas.append('{}\t{}\n'.format(text[i], 'B'))
#             num = i
#             for i in range(num + 1, num + len(target)):
#                 dev_datas.append('{}\t{}\n'.format(text[i], 'I'))
#         else:
#             dev_datas.append('{}\t{}\n'.format(text[i], 'O'))
#         i += 1
#         if i >= len(text):
#             i = 0
#             break
#         # if i<start_index:
#         #     dev_datas.append('{}\t{}\n'.format(text[i], 'O'))
#         # elif i==start_index:
#         #     dev_datas.append('{}\t{}\n'.format(text[i], 'B'))
#         # elif i>start_index and i<end_index:
#         #     dev_datas.append('{}\t{}\n'.format(text[i], 'I'))
#         # else:
#         #     dev_datas.append('{}\t{}\n'.format(text[i], 'O'))
#     dev_datas.append('\n')
#
# train_f = open(train_save_path, 'w', encoding='utf-8')
# dev_f = open(dev_save_path, 'w', encoding='utf-8')
# train_f.writelines(train_datas)
# train_f.close()
# dev_f.writelines(dev_datas)
# dev_f.close()


def get_data():
    v = 'v2'

    path = r'C:\workspace\data\qirui\defect\{}\all.jsonl'.format(v)
    import json

    labeleds = []
    un_labeleds = []

    f = open(path, 'r', encoding='utf-8')
    lines = f.readlines()
    for line in lines:
        line = line[:-1]
        result = json.loads(line)
        if result['entities'] != []:
            labeleds.append(result)
        else:
            un_labeleds.append(result)
    sub_index = int(len(labeleds)/10*8)

    import random
    from tqdm import tqdm
    random.shuffle(labeleds)
    random.shuffle(un_labeleds)

    train_dicts = labeleds[:sub_index]
    valid_dicts = labeleds[sub_index:]
    test_dits = un_labeleds

    train_file = open(r'C:\workspace\data\qirui\defect\{}\train.conll'.format(v), 'w', encoding='utf-8')
    valid_file = open(r'C:\workspace\data\qirui\defect\{}\valid.conll'.format(v), 'w', encoding='utf-8')
    test_file = open(r'C:\workspace\data\qirui\defect\{}\test.conll'.format(v), 'w', encoding='utf-8')

    for data in tqdm(train_dicts):
        text = data['text']
        while '\n' in text:
            text = text.replace('\n', ' ')
        tuple_index = []
        for index_data in data['entities']:
            tuple_index.append(index_data['start_offset'])
            tuple_index.append(index_data['end_offset'])
        i = 0
        while i< len(text):
            if i not in tuple_index:
                train_file.write('{}\t{}\n'.format(text[i], 'O'))
                i+=1
            else:
                train_file.write('{}\t{}\n'.format(text[i], 'B-defect'))
                i+=1
                while i < len(text):
                    if i not in tuple_index:
                        train_file.write('{}\t{}\n'.format(text[i], 'I-defect'))
                        i+=1
                    else:
                        i+=1
                        break
        train_file.write('\n')

    for data in tqdm(valid_dicts):
        text = data['text']
        while '\n' in text:
            text = text.replace('\n', ' ')
        tuple_index = []
        for index_data in data['entities']:
            tuple_index.append(index_data['start_offset'])
            tuple_index.append(index_data['end_offset'])
        i = 0
        while i < len(text):
            if i not in tuple_index:
                valid_file.write('{}\t{}\n'.format(text[i], 'O'))
                i += 1
            else:
                valid_file.write('{}\t{}\n'.format(text[i], 'B-defect'))
                i += 1
                while i < len(text):
                    if i not in tuple_index:
                        valid_file.write('{}\t{}\n'.format(text[i], 'I-defect'))
                        i += 1
                    else:
                        i += 1
                        break
        valid_file.write('\n')

    for data in tqdm(test_dits):
        text = data['text']
        while '\n' in text:
            text = text.replace('\n', ' ')
        text = text.strip()
        test_file.write(text + '\n')


    train_file.close()
    valid_file.close()
    test_file.close()

def get_defect(text, pred):
    result = []
    i = 0
    while i < len(pred):
        if pred[i][0] == 'B':
            r = text[i]
            i += 1
            while i < len(pred) and pred[i][0] == 'I':
                r += text[i]
                i += 1
            result.append(r)
            r = ''
        else:
            i += 1
    return result

if __name__ == '__main__':
    get_data()