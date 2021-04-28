# coding:UTF-8
# author    :Just_silent
# init time :2021/4/28 11:06
# file      :near_synonym.py
# IDE       :PyCharm

import synonyms

while True:
    print('请输入词汇：', end='')
    word = input()
    print(synonyms.nearby(word))
    print('请输入句子一：', end='')
    sen1 = input()
    print('请输入句子二：', end='')
    sen2 = input()
    print('概率：', synonyms.compare(sen1, sen2, seg=True))

# sen1 = "发生历史性变革"
# sen2 = "发生了历史性变革"
# r = synonyms.compare(sen1, sen2, seg=True)
# print(r)
#
# while True:
#     print('请输入句子一：', end='')
#     sen1 = input()
#     print('请输入句子二：', end='')
#     sen2 = input()
#     print('概率：', synonyms.compare(sen1, sen2, seg=True))