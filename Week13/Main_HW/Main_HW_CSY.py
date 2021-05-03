# 파이썬 버전이 어떤 것이든 파이썬 3 문법인 print()를 통해 출력 가능
# argparse는 필요한 인자를 명령행 인터페이스로 쉽게 작성하도록 돕는 라이브러리
# torch 모듈 불러옴
# touch 모듈 내에 있는 nn을 불러옴
# torch.nn은 클래스로 정의됨, torch.nn.functional은 함수로 정의됨
# touch 모듈 내에 있는 optim를 불러옴
# torchvision 유명 데이터셋, 구현되어 있는 모델, 이미지 전처리 도구를 포함하고 있는 패키지로 datasets, transforms(전처리 방법들)을 불러옴
# Tensor에 정의된 거의 모든 연산 지원

from torchvision.datasets import ImageFolder
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# Training settings
# 배치 사이즈는 64로 설정
batch_size = 64


# 훈련할 데이터셋을 불러옴transform을 통해 현재 데이터를 파이토치 텐서로 변환/download는 해당 경로에 MNIST 데이터가 없으면 다운로드함
train_dataset = ImageFolder("/content/custom_dataset/train", transform=transforms.Compose([transforms.RandomCrop(100),transforms.ToTensor()]))

# 테스트할 데이터셋을 불러옴transform을 통해 현재 데이터를 파이토치 텐서로 변환
test_dataset = ImageFolder("/content/custom_dataset/test",transform=transforms.Compose([transforms.RandomCrop(100),transforms.ToTensor()]))



# Data Loader (Input Pipeline)
# 훈련할 데이터 로더/훈련 데이터셋/배치 사이즈 64/무작위 순서
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)

# 테스트할 데이터 로더/테스트 데이터셋/배치 사이즈 64/순서는 차례대로
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)


# torch.nn.Module을 상속받는 파이썬 클래스 정의
# 모델의 생성자 정의 (객체가 갖는 속성값을 초기화하는 역할, 객체가 생성될 때 자동으로 호출됨)
class InceptionA(nn.Module):

    def __init__(self, in_channels):
        # super() 함수를 통해 nn.Module의 생성자를 호출함
        # 2d convolution 연산 수행/출력 채널 16개, 커널 사이즈 1x1
        super(InceptionA, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)

        # 2d convolution 연산 수행/출력 채널 16개, 커널 사이즈 1x1
        # 2d convolution 연산 수행/입력 채널 16개, 출력 채널 24개, 커널 사이즈 5x5/padding을 이용하여 가장자리에 0을 추가
        self.branch5x5_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3dbl_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)

        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)

    # 모델 객체와 학습 데이터인 x를 받음
    def forward(self, x):
        # x를 인자로 받아 2d convolution 연산 수행
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        # 커널 사이즈를 3x3으로 해  average pooling을 함
        # 2d convolution 연산 수행
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        # 출력을 리스트로 저장
        #  tensor를 두번째 차원을 기준으로 concatenate(연결시킴)->두번째 차원 늘어남
        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)

# torch.nn.Module을 상속받는 파이썬 클래스 정의
# 모델의 생성자 정의 (객체가 갖는 속성값을 초기화하는 역할, 객체가 생성될 때 자동으로 호출됨)
class Net(nn.Module):

    def __init__(self):
        # super() 함수를 통해 nn.Module의 생성자를 호출함
        # 2d convolution 연산 수행/입력 채널 3개, 출력 채널 10개, 커널 사이즈 5
        # 2d convolution 연산 수행/입력 채널 88개, 출력 채널 20개, 커널 사이즈 5
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(88, 20, kernel_size=5)

        # incept에 입력 채널 설정
        self.incept1 = InceptionA(in_channels=10)
        self.incept2 = InceptionA(in_channels=20)

        # 2x2 필터로 최댓값 추출
        # 입력을 42592, 출력을 10으로 신경망 계층 생성(fully-connected layer)
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(42592, 10)

    # 모델 객체와 학습 데이터인 x를 받아 forward 연산하는 함수로 Net(입력 데이터) 형식으로 객체를 호출하면 자동으로 forward 연산 수행됨
    def forward(self, x):
        # size(0)은 (n, 28*28)중 n(배치사이즈)을 리턴
        # x를 convolution 연산하고 maxpooling을 한 후 활성화 함수인 relu함수를 통해 나온 결과값을 x에 저장
        # x를 인자로 받아 incept1,2에 대입
        # view 함수는 원소의 수를 유지하면서 텐서의 크기를 변경함(평평하게 만듦)
        # fully-connected layer에 x를 인자로 넘긴 결과를 x에 저장
        # softmax에 log를 취하여 계산한 값을 반환
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = self.incept1(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = self.incept2(x)
        x = x.view(in_size, -1)  # flatten the tensor
        x = self.fc(x)
        return F.log_softmax(x,dim=0)

# Net 클래스 변수 model 선언
model = Net()

# SGD(확률적 경사 하강법)는 경사 하강법의 일종이고 model.parameters()를 이용하여 parameter를 전달함. lr은 학습률이며 momentum은 SGD에 관성을 더해줌(이전 이동 값을 고려하여 일정 비율만큼 다음 값을 결정)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# 학습
def train(epoch):
    # 학습 모드로 전환
    model.train()
    # train_loader로 학습
    # data와 target 변수 설정
    # 미분을 통해 얻은 기울기가 이전에 계산된 기울기 값에 누적되기 때문에 기울기 값을 0으로 초기화 함
    # data를 model에 넣어 예측값 output을 도출
    # NLL 손실함수를 이요하여 손실값 도출
    # 역전파 실행, 손실 함수를 미분하여 기울기 계산
    # step() 함수를 호출하여 parameter를 업데이트함
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        # 배치 사이즈가 10의 배수일때마다 학습을 반복한 수 loss값 출력
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data))

# 테스트 과정
def test():
    # 평가 모드로 전환
    # 정확도 계산을 위한 변수 설정
    model.eval()
    
    test_loss = 0
    correct = 0
    # test_loader로 모델 시험
    # data와 target 변수 설정/volatile=True -> (테스트 시에는)기울기를 계산하지 않음(backpropagation 안 함)
    # data를 model에 넣어 예측값 output을 도출
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        # sum up batch loss
        # 배치 손실값의 합
        test_loss += F.nll_loss(output, target, reduction='sum').data
        # get the index of the max log-probability
        # 최대값의 index를 반환/keepdim은 벡터 차원을 유지 시킬건지 아닌지를 설정
        # pred값과 target값을 비교하여 일치하는지 검사한 후 일치하는 것들의 개수의 합을 저장
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    # 테스트 손실값을 test_loader.dataset길이로 나눔
    # 테스트에 대한 손실값과 정확도 출력
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

# 10번 반복
# 훈련 후 테스트 실행
for epoch in range(1, 10):
    train(epoch)
    test()
