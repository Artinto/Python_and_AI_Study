from __future__ import print_function
import argparse ## Script로 실행할 때 인자값에 따라 동작을 다르게 하기 위한 Module 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# Training settings
batch_size = 64

# MNIST Dataset
train_dataset = datasets.MNIST(root='./data',train=True,transform=transforms.ToTensor(),download=True)
test_dataset = datasets.MNIST(root='./data',train=False,transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_loader  = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # Conv2d(input channel, output channel, kernel size/filter size)
        # Convolutional Layer. In channel = 1, Out channel = 10, filter의 사이즈는 5x5
        # Default value => padding = 0 , stride = 1 
        
        # 입력 크기 : 1 x 28 x 28 
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)   
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # 출력 Channel = 20
        
        self.mp = nn.MaxPool2d(2) 
        # Max Pooling kernel size = 2x2
        # Default value => padding = 0 , stride = 1 
        
        self.fc = nn.Linear(320, 10) # 공간 데이터를 Flatten -> Dense Net 

    def forward(self, x):
        in_size = x.size(0) # 들어온 data의 개수 -> data.shape = (n, 1, 28, 28) -> n = batch size 
        x = F.relu(self.mp(self.conv1(x))) # Convolution 수행 -> Maxpool -> relu activation function 1
        x = F.relu(self.mp(self.conv2(x))) # Convolution 수행 후 Maxpool -> relu activation function 2
        x = x.view(in_size, -1)  # flatten the tensor
        x = self.fc(x) # data를 flatten 한 후 Dense Net에 넣어줌 
        return F.log_softmax(x) # Calc Loss


model = Net()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


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
                100. * batch_idx / len(train_loader), loss.item()))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        # volatile -> 변수를 CPU Register가 아니라 Memory에 저장한다.
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).data
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1] # One - Hot 과 유사한 개념 . 가장 큰 Class의 Index
        correct += pred.eq(target.data.view_as(pred)).cpu().sum() # Target 과 pred 비교 후 맞은 개수 누적합

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, 10):
    train(epoch)
    test()
