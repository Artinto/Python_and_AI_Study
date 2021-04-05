from __future__ import print_function #파이썬 3의 문법을 파이썬 2에 사용할 수 있음
import argparse #도움말과 사용법 메시지를 자동 생성하고 사용자가 프로그램에 잘못된 인자를 줄 떄 에러 발생
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable #tensor와 비슷한 성능, 변수로 지정?
 
# Training settings
batch_size = 64 #배치 사이즈
 
# MNIST Dataset
train_dataset = datasets.MNIST(root='./data/', #학습 데이터셋 
                               train=True, 
                               transform=transforms.ToTensor(), 
                               download=True)

test_dataset = datasets.MNIST(root='./data/', #테스트 데이터셋 
                              train=False,
                              transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


class InceptionA(nn.Module):

    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=1) #1x1 아웃풋 16, kernel_size:window size

        self.branch5x5_1 = nn.Conv2d(in_channels, 16, kernel_size=1) #1x1 아웃풋 16,
        self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2) #5x5필터링,padding:주변을 숫자로 둘러주는 것(같은 사이즈 유지를 위해),  input이 16, output이 24

        self.branch3x3dbl_1 = nn.Conv2d(in_channels, 16, kernel_size=1) #1x1 아웃풋 16,
        self.branch3x3dbl_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1) #input이 16, output이 24, 3*3
        self.branch3x3dbl_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)# output이 24, 3*3

        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)
 #conv2d:filter로 특징을 뽑아내는 레이어
    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1) #3*3필터, stride=1이므로 한칸식 이동
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool] 
        return torch.cat(outputs, 1) #텐서들을 연결 dim=1은 두번째 차원을 늘리라는 것 즉


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) #합성곱 연산, input=1,outpiut=10, 필터=5*5 
        self.conv2 = nn.Conv2d(88, 20, kernel_size=5) #합성곱 연산,input=88, ouput=20,   필터=5*5, 16+24+24+24

        self.incept1 = InceptionA(in_channels=10)
        self.incept2 = InceptionA(in_channels=20)

        self.mp = nn.MaxPool2d(2) #2*2사이즈의 필터사용, 최댓값을 뽑아낸다, stride=1
        self.fc = nn.Linear(1408, 10)

    def forward(self, x):
        in_size = x.size(0)#(a, b)중 a를 받음
        x = F.relu(self.mp(self.conv1(x)))
        x = self.incept1(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = self.incept2(x)
        x = x.view(in_size, -1)  # 일자로 핌
        x = self.fc(x)
        return F.log_softmax(x)


model = Net()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)#학습률 0.01


def train(epoch):
    model.train() #학습
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target) #data를 입력과 결과값으로 분리
        optimizer.zero_grad()#미분값 초기화
        output = model(data) #예상값
        loss = F.nll_loss(output, target) #loss계산, softmax는 이미 해줌
        loss.backward()#역전파 실행  
        optimizer.step()# 파라미터 이동
        if batch_idx % 10 == 0: #loss값 
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))


def test():
    model.eval() #eval 모드에서 사용
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target) #data를 입력과 결과값
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).data[0]
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1] #예측값 output의 최댓값을 pred에
        correct += pred.eq(target.data.view_as(pred)).cpu().sum() # 맞은갯수 계산

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), #accuracy 출력
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, 10):
    train(epoch)
    test()
