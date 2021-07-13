import torch
import numpy as np


def get_attn_subsequence_mask(seq):
    '''
    主要作用是屏蔽未来时刻单词的信息
    seq: [batch_size, tgt_len]
    '''
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1) # Upper triangular matrix
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask # [batch_size, tgt_len, tgt_len]


if __name__ == '__main__':
    print(get_attn_subsequence_mask(torch.Tensor(2, 3)))