from __future__ import print_function #파이썬2, 3에서 print함수 호환
import argparse #커맨드 라인에서 인자를 입력받고 파싱, 예외처리등을 자동으로 처리 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable #텐서와 동일한 동작, autograd를 자동으로 계산

# Training settings
batch_size = 64 #배치 설정

# MNIST Dataset
train_dataset = datasets.MNIST(root='./data/', #학습 데이터셋 설정
                               train=True,
                               transform=transforms.ToTensor(), #mnist데이터를 텐서로변환
                               download=True)

test_dataset = datasets.MNIST(root='./data/', #테스트 데이터셋 설정
                              train=False,
                              transform=transforms.ToTensor()) #mnist데이터를 텐서로변환

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, #트레인로더와 테스트로더를 설정하여
                                           batch_size=batch_size, #데이터를 불러온다
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) #합성곱 연산 사용, 입력 채널=1, 아웃 채널=10. 필터=5*5 
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) #입력 채널=10, 아웃 채널=20. 필터=5*5, conv1의 아웃과 conv2의 인풋은 같아야함
        self.mp = nn.MaxPool2d(2) #2*2사이즈의 필터사용, 최댓값을 뽑아낸다, stride=1
        self.fc = nn.Linear(320, 10) 

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x))) #(1*28*28)mnist -> conv1 -> 10*24*24 -> mp -> 10*12*12 -> conv2 -> 20*8*8 -> mp -> 20*4*4 -> linear입력 320
        x = F.relu(self.mp(self.conv2(x)))
        x = x.view(in_size, -1)  # linear에 맞게 텐서 모양 변환
        x = self.fc(x)
        return F.log_softmax(x) #결과를 sofrtmax


model = Net()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    model.train() #학습중 임을 명시
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target) #data를 입력과 결과값으로 분리
        optimizer.zero_grad()
        output = model(data) #예상결과값
        loss = F.nll_loss(output, target) #loss계산
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0: #loss값 s
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test():
    model.eval() #eval 모드에서 사용한다고 선언
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target) #data를 입력과 결과값으로 분리
        output = model(data) #예상결과값
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).data #loss값의 합
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1] #예측값 output의 최댓값의 인덱스
        correct += pred.eq(target.data.view_as(pred)).cpu().sum() #결과값과 비교하여 맞은갯수 계산

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), #accuracy 출력
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, 10):
    train(epoch)
    test()
