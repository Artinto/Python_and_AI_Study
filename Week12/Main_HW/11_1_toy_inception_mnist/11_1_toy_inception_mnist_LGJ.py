# https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import print_function
import argparse # 필요한 명령행 인자를 파 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# Training settings
batch_size = 64

# MNIST Dataset
train_dataset = datasets.MNIST(root='./data/', # datasets를 통하여 MNIST데이터를 불러옴
                               train=True,  # 훈련 데이터를 반환받음
                               transform=transforms.ToTensor(), # 현재 데이터를 파이토치 텐서로 변환
                               download=True) # 해당 경로에 MNIST데이터가 없다면 다운로드 받음

test_dataset = datasets.MNIST(root='./data/', # datasets를 통하여 MNIST데이터를 불러
                              train=False, # 테스트 데이터를 반환받음
                              transform=transforms.ToTensor()) # 현재 데이터를 파이토치 텐서로 변환

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, # dataloader를 사용하여 train_dataset를 불러오고, 64의 배치사이즈 지정 및 셔플 활성화
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, # dataloader를 사용하여 test_dataset를 불러오고, 64의 배치사이즈 지정 및 셔플 비활성화
                                          batch_size=batch_size,
                                          shuffle=False)


class InceptionA(nn.Module): # 인셉션

    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=1) # (1x1)커널을 사용하여 in_channels에서 16개의 채널을 생성

        self.branch5x5_1 = nn.Conv2d(in_channels, 16, kernel_size=1) # (1x1)커널을 사용하여 in_channels에서 16개의 채널을 생성
        self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2) # (5x5)커널을 사용하여 16개의 채널에서 24개의 채널을 생성, 패딩은 바깥으로 2픽셀

        self.branch3x3dbl_1 = nn.Conv2d(in_channels, 16, kernel_size=1) # (1x1)커널을 사용하여 in_channels에서 16개의 채널을 생성
        self.branch3x3dbl_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1) # (3x3)커널을 사용하여 16개의 채널에서 24개의 채널을 생성, 패딩은 바깥으로 1픽셀
        self.branch3x3dbl_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1) # (3x3)커널을 사용하여 24개의 채널에서 24개의 채널을 생성, 패딩은 바깥으로 1픽셀

        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x) # Conv2d함수를 시행

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool] # 최종 아웃풋 리스트
        return torch.cat(outputs, 1) # 텐서를 합침


class Net(nn.Module): # nn.Module의 상속을 받는 신경망 클래스 Net 작성

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) # 입력(색)은 1개, 출력은 10개, 커널(필터) 사이즈는 5x5
        self.conv2 = nn.Conv2d(88, 20, kernel_size=5) # 입력은 10개(conv1의 출력과 같음), 출력은 20개, 커널 사이즈는 5x5

        self.incept1 = InceptionA(in_channels=10)
        self.incept2 = InceptionA(in_channels=20)

        self.mp = nn.MaxPool2d(2) # MaxPool 연산을 사용하여 최댓값만 뽑아냄
        self.fc = nn.Linear(1408, 10)  # 입력 1408, 최종 출력 10의 선형함수

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x))) # 합성곱 신경망 conv1을 maxpool을 적용하고, relu함수로 돌림
        x = self.incept1(x)
        x = F.relu(self.mp(self.conv2(x))) # 합성곱 신경망 conv2을 maxpool을 적용하고, relu함수로 돌림
        x = self.incept2(x)
        x = x.view(in_size, -1)  # flatten the tensor
        x = self.fc(x) # 선형함수를 적용
        return F.log_softmax(x) # NLLLoss를 사용하므로 마지막에 logsoftmax를 적용


model = Net() # model에Net 클래스 적용

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5) # 최적화 함수로 경사하강법을 실행, 학습률 0.01, 모멘텀(관성) 0.5로 설정


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader): # train_loader를 이용하여 각각의 배치사이즈의 data, target를 불러옴
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target) # NLL(Negative Log-Likehood)Loss를 사용하여 학습값과 참값을 비교함.
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader: # test_loader를 이용하여 각각의 배치사이즈의 data, target를 불러옴
        data, target = Variable(data, volatile=True), Variable(target) # volatile: 변수를 메모리에 저장한다.
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).data[0]
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1] # 가장 큰 클래스의 인덱스값으로 예측한다.
        correct += pred.eq(target.data.view_as(pred)).cpu().sum() # 예측값과 타겟 데이터를 비교하여 얼마나 옳았는지 합을 계산

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset))) # 예측이 얼마나 맞았는지 Accuracy 출력


for epoch in range(1, 10):
    train(epoch)
    test()
