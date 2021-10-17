# _*_ coding : utf-8 _*_
# @Time : 2021/10/10 20:19
# @Author : wxs
# @File : HW1
# @Project :
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

import csv
import numpy as np
import os
import matplotlib.pyplot as plt

myseed = 42069  # 作用是为了其他人复现的时候更接近作者的结果
np.random.seed(myseed)
torch.manual_seed(myseed)


# 数据类的处理
class Covid19dataset(Dataset):
    def __init__(self, path, mode='train'):
        super().__init__()
        self.mode = mode

        # 读取数据
        with open(path) as file:
            data_csv = list(csv.reader(file))   # 将csv文件中获取的数据转换为列表类型
            data = np.array(data_csv[1:])[:, 1:].astype(float)  # 获取文件中的【数值数据】

        if mode == 'test':  # 由于训练集、验证集与测试集的数据有所不同（最后一列数据），在此分情况运行
            data = data[:, 0:93]    # 测试集中，由于最后一行是得预测的数据，故只有93列
            self.data = torch.FloatTensor(data)     # 将numpy类型数据转换为tensor类型
        else:
            target = data[:, -1]    # 训练集和验证集中用于对比结果的“目标”
            data = data[:, 0:93]
            train_index = []
            dev_index = []
            for i in range(data.shape[0]):     # 这个循环用于将covid.train.csv文件中的数据分为训练集和验证集
                if i % 10 != 0:                # 取序号为整十数的样本作为验证集
                    train_index.append(i)
                else:
                    dev_index.append(i)
            if mode == 'train':     # 训练集的数据
                self.target = torch.FloatTensor(target[train_index])
                self.data = torch.FloatTensor(data[train_index, 0:93])
            else:       # 测试集的数据
                self.target = torch.FloatTensor(target[dev_index])
                self.data = torch.FloatTensor(data[dev_index, 0:93])

        # 此处是对数据进行标准化处理，可以将不同量纲的不同特征，变为同一个数量级，使得损失函数更加平滑
        # 标准化的优点：①提升模型的精度   ②提升收敛速度
        # 采用均值标准化： （第i维数据 - 第i维数据的平均值）/（第i维数据的标准差）
        self.data[:, 40:] = (self.data[:, 40:] - self.data[:, 40:].mean(dim=0)) / self.data[:, 40:].std(dim=0)

        self.dim = self.data.shape[1]   # 获取数据的列数

    def __getitem__(self, item):        # 【注】这是Dataset必须重写的类函数，意在按索引返回数据
        if self.mode == 'train' or self.mode == 'dev':  # 训练集和验证集包含特征和目标数据
            return self.data[item], self.target[item]
        else:
            return self.data[item]      # 测试集仅含有特征数据

    def __len__(self):                  # 【注】返回数据的行数
        return len(self.data)


def prep_dataloader(path, mode, batch_size, n_jobs=0):
    dataset = Covid19dataset(path, mode)    # 定义一个Dataset类
    dataloader = DataLoader(dataset, batch_size,
                            shuffle=(mode == 'train'),  # 【是否打乱数据后再读取】
                            drop_last=False,            # 【False表示不丢弃不能被batch_size整除的部分】
                            num_workers=n_jobs,         # 【采用多少个进程读取数据】
                            pin_memory=False)           # 【是否将数据载入CUDA的内存当中】
    print(mode, 'data done!')
    return dataloader


# 这部分可以直接到pytorch的官方文档中查看对应类或函数的使用方法
class Mymodel(nn.Module):
    def __init__(self, input_dim):
        super(Mymodel, self).__init__()
        self.net = nn.Sequential(           # 设置好模型的结构，可在forward函数中直接按顺序运行
            nn.Linear(input_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )

        self.criterion = nn.MSELoss(reduction='mean')   # 计算Loss

    def forward(self, x):
        return self.net(x).squeeze(1)       # 对数据的维度进行压缩，方便预测值与实际值的对比

    def cal_loss(self, pred, target):
        return self.criterion(pred, target)     # 计算Loss


def train(model, train_data, dev_data):
    max_epoch = 3000    # 至多训练次数
    epoch = 1
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)   # 初始化梯度下降法
    train_loss = []     # 存储训练集的Loss
    dev_loss = []       # 存储测试集的Loss
    min_mse = 1000
    break_flag = 0
    while epoch < max_epoch:
        model.train()               # 设置模式
        for x, y in train_data:     # x，y 每次包含一个batch_size的样本
            optimizer.zero_grad()   # 每次必须先将梯度清零
            pred = model(x)
            loss = model.cal_loss(pred, y)  # 计算Loss
            train_loss.append(loss.detach())
            loss.backward()         # 计算梯度
            optimizer.step()        # 模型的参数更新

        dev_mse = dev(model, dev_data)
        if dev_mse < min_mse:       # 如果测试验证集的Loss比上一次小，则存储当前模型
            min_mse = dev_mse
            print('Saving model (epoch = {:4d}, loss = {:.4f})'
                  .format(epoch + 1, min_mse))

            # 存储当前最好的模型，此处需要导入os库，创建相应的目录
            torch.save(model.state_dict(), 'my_models/mymodel.pth')

            break_flag = 0
        else:
            break_flag += 1

        dev_loss.append(dev_mse.detach())

        if break_flag > 200:    # 如果连续200个周期，Loss都没有下降，则结束训练
            break

        epoch += 1
    return train_loss, dev_loss


def dev(model, dev_data):
    model.eval()            # 设置模式
    total_loss = []

    for x, y in dev_data:       # 得到当前模型下验证集的Loss
        pred = model(x)
        dev_loss = model.cal_loss(pred, y)
        total_loss.append(dev_loss)

    return sum(total_loss) / len(total_loss)    # 取Loss的平均数


def test(model, test_data):
    model.eval()            # 设置模式
    preds = []
    for x in test_data:     # 计算获得【预测值】
        pred = model(x)
        preds.append(pred.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()
    return preds


def plot_learning_curve(train_loss, dev_loss, title=''):
    total_steps = len(train_loss)
    x_1 = range(total_steps)
    x_2 = x_1[::len(train_loss) // len(dev_loss)]
    plt.figure(1, figsize=(6, 4))
    plt.plot(x_1, train_loss, c='tab:red', label='train')
    plt.plot(x_2, dev_loss, c='tab:cyan', label='dev')
    plt.ylim(0.0, 5.)
    plt.xlabel('Training steps')
    plt.ylabel('MSE loss')
    plt.title('Learning curve of {}'.format(title))
    plt.legend()
    plt.show()


def plot_pred(dv_set, model, device, lim=35., preds=None, targets=None):
    if preds is None or targets is None:
        model.eval()
        preds, targets = [], []
        for x, y in dv_set:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                preds.append(pred.detach().cpu())
                targets.append(y.detach().cpu())
        preds = torch.cat(preds, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()
    plt.figure(2, figsize=(5, 5))
    plt.scatter(targets, preds, c='r', alpha=0.5)
    plt.plot([-0.2, lim], [-0.2, lim], c='b')
    plt.xlim(-0.2, lim)
    plt.ylim(-0.2, lim)
    plt.xlabel('ground truth value')
    plt.ylabel('predicted value')
    plt.title('Ground Truth v.s. Prediction')
    plt.show()


def save_pred(preds, file):
    print('Saving results to {}'.format(file))
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):   # enumerate() 函数用于将一个可遍历的数据对象组合为一个索引序列，同时列出数据和数据下标
            writer.writerow([i, p])

# 设置存储模型的目录
os.makedirs('my_models', exist_ok=True)

# 加载数据
train_data = prep_dataloader('covid.train.csv', 'train', batch_size=135)
dev_data = prep_dataloader('covid.train.csv', 'dev', batch_size=135)
test_data = prep_dataloader('covid.test.csv', 'test', batch_size=135)

# 设置模型与训练
mymodel = Mymodel(train_data.dataset.dim)
train_loss, dev_loss = train(mymodel, train_data, dev_data)
plot_learning_curve(train_loss, dev_loss, title='deep model')
del mymodel

# 加载最好的模型进行预测
model = Mymodel(train_data.dataset.dim)
ckpt = torch.load('my_models/mymodel.pth', map_location='cpu')  # 加载最好的模型
model.load_state_dict(ckpt)
plot_pred(dev_data, model, 'cpu')
preds = test(model, test_data)

# 储存预测结果
save_pred(preds, 'mypred.csv')
print('All Done!')