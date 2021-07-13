# import torch
# import torch.nn.functional as F
# print(torch.__version__)
# print(torch.cuda.is_available())


# cross_entropy
# input = torch.randn(3, 5, requires_grad=True)
# target = torch.randint(5, (3,), dtype=torch.int64)
# loss = F.cross_entropy(input, target)

# tensorflow_gpu暂时不支持cuda10.2
# import tensorflow as tf
# print(tf.test.is_gpu_available())

# 获取.yml文件的路径
# import os
# source_path = ''
# for p in os.listdir('./'):
#     if '.yml' in p:
#         source_path ='./' + p
# print(source_path)