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
                              transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, #트레인로더와 테스트로더를 설정하여
                                           batch_size=batch_size, #데이터를 불러온다
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


class InceptionA(nn.Module):

    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=1) #1x1필터사용, in_channels -> 아웃풋 16으로 만듬

        self.branch5x5_1 = nn.Conv2d(in_channels, 16, kernel_size=1) #1x1필터사용, in_channels -> 아웃풋 16으로 만듬
        self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2) #1x1필터을 통과한 값을 5x5필터링, padding=2로 하여 같은 사이즈 유지 

        self.branch3x3dbl_1 = nn.Conv2d(in_channels, 16, kernel_size=1) #1x1필터사용, in_channels -> 아웃풋 16으로 만듬
        self.branch3x3dbl_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1) #1x1필터을 통과한 값을 3x3필터링, padding=1로 하여 같은 사이즈 유지
        self.branch3x3dbl_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)

        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1) #3x3의 평균필터
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool] 
        return torch.cat(outputs, 1) #텐서들을 연결 dim=1, 두번쨰 차원을 늘림


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) #합성곱 연산 사용, 입력 채널=1, 아웃 채널=10. 필터=5*5 
        self.conv2 = nn.Conv2d(88, 20, kernel_size=5) #합성곱 연산 사용, 입력 채널=88, 아웃 채널=10. 필터=5*5 

        self.incept1 = InceptionA(in_channels=10)
        self.incept2 = InceptionA(in_channels=20)

        self.mp = nn.MaxPool2d(2) #2*2사이즈의 필터사용, 최댓값을 뽑아낸다, stride=1
        self.fc = nn.Linear(1408, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = self.incept1(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = self.incept2(x)
        x = x.view(in_size, -1)  # flatten the tensor
        x = self.fc(x)
        return F.log_softmax(x)


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
        if batch_idx % 10 == 0: #loss값 출력
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))


def test():
    model.eval() #eval 모드에서 사용한다고 선언
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target) #data를 입력과 결과값으로 분리
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).data[0]
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
