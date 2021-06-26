import itertools
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

import torch.nn as NN



class FashionMNISTDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        data = pd.read_csv(csv_file)
        self.X = np.array(data.iloc[:, 1:]).reshape(-1, 1, 28, 28).astype(float)
        self.Y = np.array(data.iloc[:, 0]);
        del data;  # 结束data对数据的引用,节省空间
        self.len = len(self.X)

    def __len__(self):
        # return len(self.X)
        return self.len

    def __getitem__(self, idx):
        item = self.X[idx]
        label = self.Y[idx]
        return (item, label)


DATA_PATH = Path('.')
BATCH_SIZE = 256

train_dataset = FashionMNISTDataset(csv_file=DATA_PATH / "fashion-mnist_train.csv")
test_dataset = FashionMNISTDataset(csv_file=DATA_PATH / "fashion-mnist_test_data.csv")

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=False)

a=iter(test_loader)
data=next(a)
img=data[0][4].reshape(28,28)
data[0][4].shape,img.shape

plt.imshow(img,cmap = plt.cm.gray)
plt.show()



class CNN(NN.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = NN.Sequential(
            NN.Conv2d(1, 16, kernel_size=5, padding=2),
            NN.BatchNorm2d(16),
            NN.ReLU())  # 16, 28, 28
        self.pool1 = NN.MaxPool2d(2)  # 16, 14, 14
        self.layer2 = NN.Sequential(
            NN.Conv2d(16, 32, kernel_size=3),
            NN.BatchNorm2d(32),
            NN.ReLU())  # 32, 12, 12
        self.layer3 = NN.Sequential(
            NN.Conv2d(32, 64, kernel_size=3),
            NN.BatchNorm2d(64),
            NN.ReLU())  # 64, 10, 10
        self.pool2 = NN.MaxPool2d(2)  # 64, 5, 5
        self.fc = NN.Linear(5 * 5 * 64, 10)

    def forward(self, x):
        out = self.layer1(x)
        # print(out.shape)
        out = self.pool1(out)
        # print(out.shape)
        out = self.layer2(out)
        # print(out.shape)
        out = self.layer3(out)
        # print(out.shape)
        out = self.pool2(out)
        # print(out.shape)
        out = out.view(out.size(0), -1)
        # print(out.shape)
        out = self.fc(out)
        return out


cnn = CNN();
# 可以通过以下方式验证，没报错说明没问题，
# cnn(torch.rand(1,1,28,28))

# 打印下网络，做最后的确认
# print(cnn)

DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
print(DEVICE)

# 先把网络放到cpu上
cnn = cnn.to(DEVICE)
# 损失函数也需要放到CPU中
criterion = NN.CrossEntropyLoss().to(DEVICE)

# 另外一个超参数，学习率
LEARNING_RATE = 0.01
# 优化器不需要放GPU
optimizer = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)

# 另外一个超参数，指定训练批次
TOTAL_EPOCHS = 50


# 记录损失函数
losses = [];
for epoch in range(TOTAL_EPOCHS):
    for i, (images, labels) in enumerate(train_loader):
        images = images.float().to(DEVICE)
        labels = labels.to(DEVICE)
        # 清零
        optimizer.zero_grad()
        outputs = cnn(images)
        # 计算损失函数
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu().data.item());
        if (i + 1) % 100 == 0:
            print('Epoch : %d/%d, Iter : %d/%d,  Loss: %.4f' % (
                epoch + 1, TOTAL_EPOCHS, i + 1, len(train_dataset) // BATCH_SIZE, loss.data.item()))

plt.xkcd();
plt.xlabel('Epoch #');
plt.ylabel('Loss');
plt.plot(losses);
plt.show();


torch.save(cnn.state_dict(), "fm-cnn3.pth")
# 加载用这个
# cnn.load_state_dict(torch.load("fm-cnn3.pth"))


cnn.eval()
correct = 0
list = []
total = 0
for images, labels in test_loader:
    images = images.float().to(DEVICE)
    outputs = cnn(images).cpu()
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    # print(predicted)
    list.extend(predicted.numpy().tolist())
    #print(list)
    #print(labels)
    #correct += (predicted == labels).sum()

print(list)
name = ['one']
list = pd.DataFrame(columns=name, data=list)
list.to_csv('./testcsv.csv',encoding='utf-8')
# print('准确率: %.4f %%' % (100 * correct / total))


'''
cnn.train()
LEARNING_RATE=LEARNING_RATE / 10
TOTAL_EPOCHS=20
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)
losses = [];
for epoch in range(TOTAL_EPOCHS):
    for i, (images, labels) in enumerate(train_loader):
        images = images.float().to(DEVICE)
        labels = labels.to(DEVICE)
        #清零
        optimizer.zero_grad()
        outputs = cnn(images)
        #计算损失函数
        #损失函数直接放到CPU中，因为还有其他的计算
        loss = criterion(outputs, labels).cpu()
        loss.backward()
        optimizer.step()
        losses.append(loss.data.item());
        if (i+1) % 100 == 0:
            print ('Epoch : %d/%d, Iter : %d/%d,  Loss: %.4f'%(epoch+1, TOTAL_EPOCHS, i+1, len(train_dataset)//BATCH_SIZE, loss.data.item()))

plt.xkcd();
plt.xlabel('Epoch #');
plt.ylabel('Loss');
plt.plot(losses);
plt.show();

torch.save(cnn.state_dict(), "fm-cnn3.pth")

'''