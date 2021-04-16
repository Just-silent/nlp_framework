# coding:UTF-8
# author    :Just_silent
# init time :2021/4/9 15:09
# file      :utils.py
# IDE       :PyCharm


class Tool():
    def __init__(self):
        pass

    def get_tags(self, text, words):
        '''
        Args:
            text: the str of a sentence
            words : the list of a word
        Returns:
            the list of tag
        '''
        tag_list = ['O']*len(text)
        for word in words:
            s = text.find(word)
            e = s + len(word)
            tag_list[s] = 'B'
            s += 1
            while s < e:
                tag_list[s] = 'I'
                s += 1
        return tag_list

    def get_result_by_sentence_tag(self, chars, tags):
        '''
        Args:
            chars: the list of a sentence
            tags : the list of a tags
        Returns:
            the list of entity by BIO
        '''
        result = []
        i = 0
        while i<len(chars):
            if tags[i][0] == 'B':
                word = chars[i]
                while i+1<len(chars) and tags[i+1][0] == 'I':
                    word = word + chars[i+1]
                    i+=1
                result.append(word)
            i+=1
        return result
        pass

    def get_vocab_list(self, path):
        '''
        Args:
            path : the path of the vocab
        Returns:
            the list of all vocabs
        '''
        vocabs = {}
        vocab_file = open(path, 'r', encoding='utf-8')
        i = 1
        for line in vocab_file.readlines():
            vocabs[line.strip()] = i
            i+=1
        return vocabs