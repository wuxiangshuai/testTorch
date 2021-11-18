# _*_ coding : utf-8 _*_
# @Time : 2021/11/17 20:00
# @Author : wxs
# @File : showMaxPool2d
# @Project :
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

# ps 这里用灰度图的目的是简化操作，因为如果是RGB 这需要转换成pytorch的数据格式(batch, channel, H, W)
# 而 im = Image.open('data/cat.png').convert('RGB') ---> im 的shape为 (H, W, 3)

im = Image.open('test.jpg').convert('L') # 读入一张灰度图的图片
im = np.array(im, dtype='float32') # 将其转换为一个矩阵

plt.imshow(im.astype('uint8'), cmap='gray')
plt.show() # 可视化图片

# 将图片矩阵转化为 pytorch tensor，并适配卷积输入的要求
im = torch.from_numpy(im.reshape((1, 1, im.shape[0], im.shape[1])))


def F_max_pool2d(): # 使用 nn.MaxPool2d
    print('before max pool, image shape: {} x {}'.format(im.shape[2], im.shape[3]))
    small_im2 = F.max_pool2d(im, 2, 2)
    small_im2 = small_im2.data.squeeze().numpy()
    print('after max pool, image shape: {} x {} '.format(small_im2.shape[0], small_im2.shape[1]))

    plt.imshow(small_im2, cmap='gray')
    plt.show()

F_max_pool2d()