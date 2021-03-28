from __future__ import print_function # 파이썬2, 3 어떤 버전이든 파이썬3의 print 사용가능
import argparse # 명령행의 인자를 파싱
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# torchvision : 이미지 변환을 위해
# datasets : for using MNIST
# transforms : 다양한 이미지 변환 제공 (현재 과제에서는 numpy -> tensor)
from torchvision import datasets, transforms
from torch.autograd import Variable

# Training settings
batch_size = 64

# MNIST Dataset
train_dataset = datasets.MNIST(root='./data/', # data path
                               train=True,     # specify train data
                               transform=transforms.ToTensor(), # transform numpy to tensor
                               download=True)  # if doesn't have MNIST, download MNIST

test_dataset = datasets.MNIST(root='./data/', # data path
                              train=False,    # specify test data
                              transform=transforms.ToTensor()) # transform numpy to tensor

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, # use train dataset
                                           batch_size=batch_size, # batch size = 64
                                           shuffle=True) # shuffle data

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,  # use test dataset 사용
                                          batch_size=batch_size, # batch size = 64
                                          shuffle=False) # don't shuffle


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # Conv2d(input image의 채널 수, convolution에 의해 생성된 채널 수, filter size)
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) # conv1에서 생성된 채널이 conv2의 input이다
        self.mp = nn.MaxPool2d(2)    # filter size = 2*2 이고 거기서 max값을 뽑음
        self.fc = nn.Linear(320, 10) # define fully connect layer (input = 320, output = 10)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x))) # conv1(x)의 결과값을 maxpolling하고 relu 진행
        x = F.relu(self.mp(self.conv2(x))) # conv2(위 행의 결과값)의 결과값을 maxpolling하고 relu 진행
        x = x.view(in_size, -1)  # flatten the tensor, shape 변환
        x = self.fc(x) # fully connect layer
        return F.log_softmax(x) # return x의 log_softmax한 값


model = Net()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch): # training train set
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
                100. * batch_idx / len(train_loader), loss.item()))


def test(): # start test
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).data
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