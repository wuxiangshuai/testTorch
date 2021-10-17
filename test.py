# _*_ coding : utf-8 _*_
# @Time : 2021/10/9 19:48
# @Author : wxs
# @File : test
# @Project :
import torch
# Tensor 高维矩阵
# 降维
# x = torch.zeros([1, 2, 3])
# x.shape
# print(x)
# x = x.squeeze(0)
# x.shape
# print(x)

# 升维
# x = torch.zeros([2, 3])
# x.shape
# print(x)
# x = x.unsqueeze(1)
# x.shape
# print(x)

# 对调
# x = torch.zeros([2, 3])
# x.shape
# print(x)
# x = x.transpose(0, 1)
# x.shape
# print(x)

# 拼接
# x = torch.zeros([2, 1, 3])
# y = torch.zeros([2, 3, 3])
# z = torch.zeros([2, 2, 3])
# w = torch.cat([x, y, z], dim=1)
# w.shape
# print(w)

# Calculate Gradient
x = torch.tensor([[1., 0.], [-1., 1.]], requires_grad=True)
z = x.pow(2).sum()
z.backward()
x = x.grad
print(x)