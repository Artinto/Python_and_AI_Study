from __future__ import print_function # python 2 의 print 함수 기능을 python 3 에서도 사용가능하도록 해주는 모듈
import argparse # 명령행 옵션, 인자와 부속 명령을 위한 파서
import torch
import torch.nn as nn
import torch.nn.functional as F # nn의 다양한 함수가 있는 functional 모듈 불러옴
import torch.optim as optim
from torchvision import datasets, transforms # datasets : 여러가지 대표적 데이터셋을 웹에서 불러오는 모듈. MNIST 데이터 로드, transforms : 다양한 이미지 변환 기능들을 제공
from torch.autograd import Variable # autograd : Tensor의 모든 연산에 대해 자동 미분을 제공, 

# Training settings
batch_size = 64 # 배치 사이즈는 64

# MNIST Dataset
train_dataset = datasets.MNIST(root='./data/', # dataset의 root directory : MNIST/processed/training.pt 와 MNIST/processed/test.pt가 존재
                               train=True, # training.pt으로부터 dataset 생성
                               transform=transforms.ToTensor(), # PIL image 를 Tensor type 으로 변환
                               download=True) # True -> 인터넷으로부터 dataset 다운로드 받고 root directory에 넣음. 
                                                # 이미 다운로드 되어 있으면 다운로드 안함

test_dataset = datasets.MNIST(root='./data/', 
                              train=False, # test.pt으로부터 dataset 생성
                              transform=transforms.ToTensor()) # PIL image 를 Tensor type 으로 변환

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, # train_loader: train_dataset 이용한 custom dataloader
                                           batch_size=batch_size,
                                           shuffle=True) # 무작위로 섞어줌

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, # test_loader: test_dataset 이용한 custom dataloader
                                          batch_size=batch_size,
                                          shuffle=False) # 섞지 않음 -> 순서대로 테스트


class Net(nn.Module): # nn.Module의 상속을 받는 신경망 클래스 작성

    def __init__(self):
        super(Net, self).__init__() # nn.Module 의 변수 가져옴
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) # in channel: 1, out channel: 10, 커널 사이즈는 5x5
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) # in channel: 10(conv1의 out channel과 같음), outchannel: 20, 커널 사이즈는 5x5
        self.mp = nn.MaxPool2d(2) # MaxPool 사용하여 커널 사이즈 2x2에서 최댓값 선택
        self.fc = nn.Linear(320, 10) # in channel 320, out channel 10의 신경망 계층 생성

    def forward(self, x): 
        in_size = x.size(0) # (n,28*28) 의 n
        x = F.relu(self.mp(self.conv1(x))) # 합성곱 신경망 conv1을 maxpool을 적용하고, relu함수 적용
        x = F.relu(self.mp(self.conv2(x))) # 합성곱 신경망 conv2을 maxpool을 적용하고, relu함수 적용
        x = x.view(in_size, -1)  # flatten the tensor
        x = self.fc(x) # 신경망 계층에 x 전달 하여 update
        return F.log_softmax(x) # 로그(소프트맥스)


model = Net() # Net 의 instance 생성

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5) # 최적화 함수로 경사하강법을 실행, 학습률 0.01, 모멘텀(관성) 0.5로 설정


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader): # train_loader를 이용하여 각각의 배치사이즈의 data, target를 불러옴
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
    for data, target in test_loader: # test_loader를 이용하여 각각의 배치사이즈의 data, target를 불러옴
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).data
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum() # 예측값과 타겟 데이터를 비교하여 얼마나 옳았는지 합을 계산

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset))) # 예측이 얼마나 맞았는지 Accuracy 출력


for epoch in range(1, 10):
    train(epoch)
    test()
