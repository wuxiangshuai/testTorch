#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
@ModuleName: predict
@Function: 加载模型，对所画的数字进行预测
@Author: PengKai
@Time: 2020/11/24 9:40
"""

import os
import torch
import numpy as np
from dNewClassfier import CNN
from PIL import Image


# 预测自己的图片
def predict_number():
    #  加载参数
    ckpt = torch.load('cnn_minist.pkl')
    model = CNN()
    model.load_state_dict(ckpt['state_dict'])  # 参数加载到指定模型cnn
    #  要识别的图片
    file_paths = os.listdir('./demo')
    with torch.no_grad():
        for file_path in file_paths:
            input_image = './demo/' + file_path
            # 读取图片数据
            im = Image.open(input_image).resize((28, 28))
            im = im.convert('L')  # 灰度图
            im_data = np.array(im)
            im_data = torch.from_numpy(im_data).float()
            im_data = im_data.view(1, 1, 28, 28)
            out = model(im_data)
            _, pred = torch.max(out, 1)
            # 输出结果
            print(input_image)
            print('预测为数字{}。'.format(pred.item()))


if __name__ == '__main__':
    predict_number()
