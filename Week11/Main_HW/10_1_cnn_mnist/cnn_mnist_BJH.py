# https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F #함수형태로 불러오기
import torch.optim as optim
from torchvision import datasets, transforms #관련 예제 불러오기
from torch.autograd import Variable

# Training settings
batch_size = 64 # 연산에 쓰일 사이즈

# MNIST Dataset
train_dataset = datasets.MNIST(root='./data/',
                               train=True,
                               transform=transforms.ToTensor(), #텐서로 변경
                               download=True)

test_dataset = datasets.MNIST(root='./data/',
                              train=False,
                              transform=transforms.ToTensor()) #텐서로 변경

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) #5x5의 필터로 1개의 채널을 10개의 채널로 만든다.
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) #10개의 채널을 20개의 채널로 만든다
        self.mp = nn.MaxPool2d(2)#2x2 맥스 샘플링을 한다
        self.fc = nn.Linear(320, 10) #Fully Connected Linear로 320

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x))) # 컨볼루션 하고 맥스풀링하고 relu로 연산
        x = F.relu(self.mp(self.conv2(x)))
        x = x.view(in_size, -1)  # flatten the tensor/ linear로 쫙 펴주기
        x = self.fc(x) #그것을 10개의 아웃풋으로 바꿔주기
        return F.log_softmax(x) #소프트맥스 적용


model = Net()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target) #crossentropyloss는 softmax를 포함하여 nll_loss를 사용
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target) #volatile=true는 기울기를 계산하지 않겠다는 의미
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).data #size_average=flase를 통하여 loss들의 평균계산 X
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1] #값에 맞는 라벨 찾아오기
        correct += pred.eq(target.data.view_as(pred)).cpu().sum() # pred.eq는 data와 pred배열과 일치하나 확인

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, 10):
    train(epoch)
    test()
