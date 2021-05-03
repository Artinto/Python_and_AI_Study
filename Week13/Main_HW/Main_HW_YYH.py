# -*- coding: utf-8 -*-
"""Untitled16.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1NYQZxYj-rcYWPOwZuraqYfBYOXdpn5jv
"""

# Commented out IPython magic to ensure Python compatibility.

from PIL import Image
# %matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import torch
import numpy as np
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

trainset = ImageFolder("/content/drive/MyDrive/custom_dataset/train",
                         transform=transforms.Compose([transforms.RandomCrop(100),
                                                       transforms.ToTensor()]))

testset = ImageFolder("/content/drive/MyDrive/custom_dataset/test",
                        transform=transforms.Compose([transforms.RandomCrop(100),
                                                      transforms.ToTensor()]))

train_loader = data.DataLoader(trainset, batch_size=32, shuffle=True)
test_loader = data.DataLoader(testset, batch_size=32, shuffle=True)


class InceptionA(nn.Module):

    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)      # 1x1 필터로 인 채널 -> 아웃 16으로 만든다

        self.branch5x5_1 = nn.Conv2d(in_channels, 16, kernel_size=1)    # 1x1 필터로 인 채널 -> 아웃 16으로 만든다
        self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)  # 5x5 필터로 인 16 -> 아웃 24으로 만든다 padding=2 로 함으로써 사이즈를 맞춤
        self.branch3x3dbl_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3dbl_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1) #사이즈를 맞추기위해 2번 진행한것으로 생각됌

        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1) #2d avg 풀링링  
        branch_pool = self.branch_pool(branch_pool)                       

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool] 
        return torch.cat(outputs, 1)#텐서들을 모두 합침   


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)    #인 1채널 -> 아웃 16으로 만든다
        self.conv2 = nn.Conv2d(88, 20, kernel_size=5)   #인 88채널 -> 아웃 20으로 만든다

        self.incept1 = InceptionA(in_channels=10)     #인셉션에 in_channel 입력
        self.incept2 = InceptionA(in_channels=20)

        self.mp = nn.MaxPool2d(2)                     #maxpooling 으로 사이즈 절반
        self.fc = nn.Linear(42592, 10)         

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = self.incept1(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = self.incept2(x)
        x = x.view(in_size, -1)  # flatten the tensor
        x = self.fc(x)
        return F.log_softmax(x,dim=0)


model = Net()

optimizer = optim.SGD(model.parameters(), lr=0.015, momentum=0.6)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, reduction='sum').data
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, 10):
    train(epoch)
    test()
    
    
    
    
    Train Epoch: 1 [0/1949 (0%)]	Loss: 3.465164
Train Epoch: 1 [320/1949 (16%)]	Loss: 3.467974
Train Epoch: 1 [640/1949 (33%)]	Loss: 3.464832
Train Epoch: 1 [960/1949 (49%)]	Loss: 3.464750
Train Epoch: 1 [1280/1949 (66%)]	Loss: 3.461742
Train Epoch: 1 [1600/1949 (82%)]	Loss: 3.433656
Train Epoch: 1 [1740/1949 (98%)]	Loss: 3.340954
/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:110: UserWarning: volatile was removed and now has no effect. Use `with torch.no_grad():` instead.

Test set: Average loss: 3.3559, Accuracy: 29/100 (29%)

Train Epoch: 2 [0/1949 (0%)]	Loss: 3.439442
Train Epoch: 2 [320/1949 (16%)]	Loss: 3.453103
Train Epoch: 2 [640/1949 (33%)]	Loss: 3.386129
Train Epoch: 2 [960/1949 (49%)]	Loss: 3.401870
Train Epoch: 2 [1280/1949 (66%)]	Loss: 3.456865
Train Epoch: 2 [1600/1949 (82%)]	Loss: 3.416420
Train Epoch: 2 [1740/1949 (98%)]	Loss: 3.346956

Test set: Average loss: 3.3257, Accuracy: 34/100 (34%)

Train Epoch: 3 [0/1949 (0%)]	Loss: 3.517059
Train Epoch: 3 [320/1949 (16%)]	Loss: 3.425750
Train Epoch: 3 [640/1949 (33%)]	Loss: 3.402493
Train Epoch: 3 [960/1949 (49%)]	Loss: 3.405547
Train Epoch: 3 [1280/1949 (66%)]	Loss: 3.420150
Train Epoch: 3 [1600/1949 (82%)]	Loss: 3.377005
Train Epoch: 3 [1740/1949 (98%)]	Loss: 3.224404

Test set: Average loss: 3.4033, Accuracy: 35/100 (35%)

Train Epoch: 4 [0/1949 (0%)]	Loss: 3.483952
Train Epoch: 4 [320/1949 (16%)]	Loss: 3.375319
Train Epoch: 4 [640/1949 (33%)]	Loss: 3.482353
Train Epoch: 4 [960/1949 (49%)]	Loss: 3.362011
Train Epoch: 4 [1280/1949 (66%)]	Loss: 3.477455
Train Epoch: 4 [1600/1949 (82%)]	Loss: 3.354793
Train Epoch: 4 [1740/1949 (98%)]	Loss: 3.289871

Test set: Average loss: 3.3355, Accuracy: 36/100 (36%)

Train Epoch: 5 [0/1949 (0%)]	Loss: 3.453651
Train Epoch: 5 [320/1949 (16%)]	Loss: 3.341218
Train Epoch: 5 [640/1949 (33%)]	Loss: 3.391204
Train Epoch: 5 [960/1949 (49%)]	Loss: 3.346833
Train Epoch: 5 [1280/1949 (66%)]	Loss: 3.383742
Train Epoch: 5 [1600/1949 (82%)]	Loss: 3.408597
Train Epoch: 5 [1740/1949 (98%)]	Loss: 3.324639

Test set: Average loss: 3.3016, Accuracy: 37/100 (37%)

Train Epoch: 6 [0/1949 (0%)]	Loss: 3.355824
Train Epoch: 6 [320/1949 (16%)]	Loss: 3.397428
Train Epoch: 6 [640/1949 (33%)]	Loss: 3.425717
Train Epoch: 6 [960/1949 (49%)]	Loss: 3.277164
Train Epoch: 6 [1280/1949 (66%)]	Loss: 3.384890
Train Epoch: 6 [1600/1949 (82%)]	Loss: 3.425402
Train Epoch: 6 [1740/1949 (98%)]	Loss: 3.337262

Test set: Average loss: 3.2640, Accuracy: 41/100 (41%)

Train Epoch: 7 [0/1949 (0%)]	Loss: 3.329717
Train Epoch: 7 [320/1949 (16%)]	Loss: 3.364171
Train Epoch: 7 [640/1949 (33%)]	Loss: 3.376855
Train Epoch: 7 [960/1949 (49%)]	Loss: 3.486858
Train Epoch: 7 [1280/1949 (66%)]	Loss: 3.427505
Train Epoch: 7 [1600/1949 (82%)]	Loss: 3.592683
Train Epoch: 7 [1740/1949 (98%)]	Loss: 3.373320

Test set: Average loss: 3.2700, Accuracy: 37/100 (37%)

Train Epoch: 8 [0/1949 (0%)]	Loss: 3.440759
Train Epoch: 8 [320/1949 (16%)]	Loss: 3.429246
Train Epoch: 8 [640/1949 (33%)]	Loss: 3.446510
Train Epoch: 8 [960/1949 (49%)]	Loss: 3.442460
Train Epoch: 8 [1280/1949 (66%)]	Loss: 3.171880
Train Epoch: 8 [1600/1949 (82%)]	Loss: 3.372365
Train Epoch: 8 [1740/1949 (98%)]	Loss: 3.279615

Test set: Average loss: 3.3819, Accuracy: 34/100 (34%)

Train Epoch: 9 [0/1949 (0%)]	Loss: 3.376528
Train Epoch: 9 [320/1949 (16%)]	Loss: 3.337636
Train Epoch: 9 [640/1949 (33%)]	Loss: 3.667301
Train Epoch: 9 [960/1949 (49%)]	Loss: 3.350829
Train Epoch: 9 [1280/1949 (66%)]	Loss: 3.364783
Train Epoch: 9 [1600/1949 (82%)]	Loss: 3.353468
Train Epoch: 9 [1740/1949 (98%)]	Loss: 3.296811

Test set: Average loss: 3.3434, Accuracy: 36/100 (36%)
