# _*_ coding : utf-8 _*_
# @Time : 2021/11/17 14:22
# @Author : wxs
# @File : dNewClassfier
# @Project :
# 1、加载必要的库
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# 2、定义超参数：用来定义模型结构和优化策略
BATCH_SIZE = 16 #每批处理的数据
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 是否用GPU训练
EPOCHS = 10 # 训练数据集的轮次

# 3、构建 pipeline，对图像处理
pipeline = transforms.Compose([
        transforms.ToTensor(), # 将图像转换成 tensor
        transforms.Normalize((0.1307,), (0.3081,)) # 正则化：降低模型复杂度以防止过拟合现象
        ])

# 4、下载、加载数据
from torch.utils.data import DataLoader
# 下载数据库
DOWNLOAD_MINIST = False
train_data = datasets.MNIST(
    train=True,
    root='./MINIST',
    transform=pipeline,
    download=DOWNLOAD_MINIST
)
test_data = datasets.MNIST(
    train=False,
    root='./MINIST',
    transform=pipeline,
    download=DOWNLOAD_MINIST
)

# 加载数据
train_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True) #shuffle=True：打乱排序
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)


# 5、构建网络模型
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # in_channels, out_channels, kernel_size, stride
        self.conv1 = nn.Conv2d(1, 10, 5, 1) # 1、灰度图片的通道， 10：输出通道， 5：kernel
        self.conv2 = nn.Conv2d(10, 20, 3, 1) # 10：输入通道， 20：输出通道你， 3：Kernel
        # self.dropout1 = nn.Dropout(0.25)
        # self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(20*10*10, 500) # 20*10*10：输入通道, 500：输出通道
        self.fc2 = nn.Linear(500, 10) # 500:输入通道， 10：输出通道

    def forward(self, x):
        input_size = x.size(0) # batch_size
        x = self.conv1(x) # 输入：batch*1*28*28，输出：batch*10*24*24(28-5+1)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2) # batch*10*24*24 -> batch*10*12*12
        x = self.conv2(x) # batch*10*12*12 -> batch*20*10*10(12-3+1)
        x = F.relu(x)
        # x = self.dropout1(x)
        x = x.view(input_size, -1) # 拉平， -1 自动计算维度：20*10*10
        x = self.fc1(x) # batch*2000 -> batch*500
        x = F.relu(x)
        # x = self.dropout2(x)
        x = self.fc2(x) # batch*500 -> batch*10
        output = F.log_softmax(x, dim=1) # 计算分类后每个数组的概率值
        return output


# 6、定义优化器
model = CNN().to(DEVICE)
optimizer = optim.Adam(model.parameters()) # 更新模型参数，使结果最优


# 7、定义训练方法
def train_model(model, device, train_loader, optimizer, epoch):
    # 训练模型
    model.train()
    train_loss = []  # 存储训练集的Loss
    train_acc = []  # 存储训练集的acc
    for batch_index, (data, target) in enumerate(train_loader):
        # 部署到Device上去
        data, target = data.to(device), target.to(device)
        # 梯度初始化为0
        optimizer.zero_grad()
        # 训练后的结果
        output = model(data)
        # 计算损失
        loss = F.cross_entropy(output, target) # 适合多分类
        # 找到概率值最大的下标
        pred = output.argmax(dim=1)
        # 反向传播：将预测值与实际值对比，而后向前更新权值、参数
        loss.backward()
        # 参数优化
        optimizer.step()
        if batch_index % 3000 == 0:
            train_loss.append(loss.item())
            print("Train Epoch : {} \t Loss : {:.6f}".format(epoch, loss.item()))
    return train_loss


# 8、定义测试方法
def test_model(model, device, test_loader):
    # 模型验证
    model.eval()
    # 正确率
    correct = 0.0
    # 测试损失
    test_loss = 0.0
    dev_loss = []
    with torch.no_grad(): # 不进行梯度下降和反向传播
        for data, target in test_loader:
            # 部署到 device 上
            data, target = data.to(device), target.to(device)
            # 测试数据
            output = model(data)
            # 计算测试损失
            test_loss += F.cross_entropy(output, target).item()
            # 找到概率值最大的下标
            pred = output.argmax(dim=1, keepdim=True) # 值，索引
            # 累计正确率
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        dev_loss.append(test_loss)
        print("Test --- Average loss : {:.4f}, Accuracy : {:.3f}\n"
              .format(test_loss, 100.0*correct/len(test_loader.dataset)))
    return dev_loss


import matplotlib.pyplot as plt


def plot_learning_curve(train_loss, dev_loss, title=''):
    total_steps = len(train_loss)
    x_1 = range(total_steps)
    x_2 = x_1[::len(train_loss) // len(dev_loss)]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x_1, train_loss, c='tab:red', label='train')
    ax2 = ax.twinx()
    ax2.plot(x_2, dev_loss, c='tab:cyan', label='dev')
    ax.grid()
    ax.legend()
    ax.set_xlabel("Training steps")
    ax.set_ylabel("MSE loss")
    ax2.set_ylabel("Dev steps")
    ax2.set_ylim(0, 0.01)
    ax.set_ylim(0, 3)
    plt.title('Learning curve of {}'.format(title))
    plt.legend()
    plt.show()
# def plot_learning_curve(train_loss, dev_loss, title=''):
#     total_steps = len(train_loss)
#     x_1 = range(total_steps)
#     x_2 = x_1[::len(train_loss) // len(dev_loss)]
#     plt.figure(1, figsize=(6, 4))
#     plt.plot(x_1, train_loss, c='tab:red', label='train')
#     plt.plot(x_2, dev_loss, c='tab:cyan', label='dev')
#     plt.ylim(0.00, 2.5)
#     plt.xlabel('Training steps')
#     plt.ylabel('MSE loss')
#     plt.title('Learning curve of {}'.format(title))
#     plt.legend()
#     plt.show()
def plot_accuracy_curve(train_acc, dev_acc, title=''):
    total_steps = len(train_acc)
    x_1 = range(total_steps)
    x_2 = x_1[::len(train_acc) // len(dev_acc)]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x_1, train_acc, c='tab:red', label='train')
    ax2 = ax.twinx()
    ax2.plot(x_2, dev_acc, c='tab:cyan', label='dev')
    ax.grid()
    ax.legend()
    ax.set_xlabel("Training steps")
    ax.set_ylabel("MSE ACC")
    ax2.set_ylabel("Dev steps")
    ax2.set_ylim(0, 0.01)
    ax.set_ylim(0, 3)
    plt.title('Learning curve of {}'.format(title))
    plt.legend()
    plt.show()



# 9、调用方法 7 、 8
train_loss = []
dev_loss = []
for epoch in range(1, EPOCHS + 1):
    trainLoss = train_model(model, DEVICE, train_loader, optimizer, epoch)
    train_loss.append(trainLoss)
    devLoss = test_model(model, DEVICE, test_loader)
    dev_loss.append(devLoss)
plot_learning_curve(train_loss, dev_loss, title='CNN model')

# torch.save(model.state_dict(), "mnist_cnn.pt")
torch.save({'state_dict': model.state_dict()}, 'cnn_minist.pkl')
print('finish training')