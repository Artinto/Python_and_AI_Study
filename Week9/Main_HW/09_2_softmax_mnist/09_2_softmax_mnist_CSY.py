# https://github.com/pytorch/examples/blob/master/mnist/main.py
# 파이썬 버전이 어떤 것이든 파이썬 3 문법인 print()를 통해 출력 가능
# touch 모듈 내에 있는 nn, optim, cuda(GPU 사용)를 불러옴
# torch.utils.data는 SGD(Stochastic Gradient Descent)의 반복 연산을 실행할 때 사용하는 미니 배치용 유틸리티 함수 포함
# torchvision 유명 데이터셋, 구현되어 있는 모델, 이미지 전처리 도구를 포함하고 있는 패키지로 datasets, transforms(전처리 방법들)을 불러옴
#  torch.nn은 클래스로 정의됨, torch.nn.functional은 함수로 정의됨
# time 모듈 불러옴
from __future__ import print_function
from torch import nn, optim, cuda
from torch.utils import data
from torchvision import datasets, transforms
import torch.nn.functional as F
import time

# Training settings
# 배치 사이즈는 64로 설정
# cuda(GPU) 사용 가능하다면 사용, 안되면 CPU 사용
# 사용 device 출력
batch_size = 64
device = 'cuda' if cuda.is_available() else 'cpu'
print(f'Training MNIST Model on {device}\n{"=" * 44}')

# MNIST Dataset
# 훈련할 데이터셋을 불러옴/train을 True로 하면 훈련 데이터 셋을 리턴받음/transform을 통해 현재 데이터를 파이토치 텐서로 변환/download는 해당 경로에 MNIST 데이터가 없으면 다운로드함
train_dataset = datasets.MNIST(root='./mnist_data/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

# 테스트할 데이터셋을 불러옴/train을 False로 하면 데스트 데이터 셋을 리턴받음/transform을 통해 현재 데이터를 파이토치 텐서로 변환
test_dataset = datasets.MNIST(root='./mnist_data/',
                              train=False,
                              transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
# 훈련할 데이터 로더/훈련 데이터셋/배치 사이즈 64/무작위 순서
train_loader = data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

# 데스트할 데이터 로더/테스트 데이터셋/배치 사이즈 64/순서는 차례대로
test_loader = data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# torch.nn.Module을 상속받는 파이썬 클래스 정의
# 모델의 생성자 정의 (객체가 갖는 속성값을 초기화하는 역할, 객체가 생성될 때 자동으로 호출됨)
class Net(nn.Module):

    def __init__(self):
        # super() 함수를 통해 nn.Module의 생성자를 호출함
        # nn.Linear() 함수를 이용하여 5개 층의 모델을 만듦((입력,출력)784과 10은 고정)
        super(Net, self).__init__()
        self.l1 = nn.Linear(784, 520)
        self.l2 = nn.Linear(520, 320)
        self.l3 = nn.Linear(320, 240)
        self.l4 = nn.Linear(240, 120)
        self.l5 = nn.Linear(120, 10)

    # 모델 객체와 학습 데이터인 x를 받아 forward 연산하는 함수로 Net(입력 데이터) 형식으로 객체를 호출하면 자동으로 forward 연산 수행됨
    def forward(self, x):
        # view 함수는 원소의 수를 유지하면서 텐서의 크기를 변경함/1 x 28 x 28 벡터(28 x 28 크기, 1가지 색)를 784 길이만큼으로 변경/-1은 첫번째 차원은 파이토치에 맡겨 설정한다는 의미/784는 두번째 차원의 길이를 의미
        # x를 선형함수에 넣고 활성화 함수인 relu함수를 통해 나온 결과값을 x에 저장
        # 마지막 return 값은 활성화 함수 사용하지 않음(logits을 사용하기 때문)
        x = x.view(-1, 784)  # Flatten the data (n, 1, 28, 28)-> (n, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)

# Net 클래스 변수 model 선언
# device에 모델 등록
# 손실함수 정의 코드로, CrossEntropyLoss 함수 사용
# SGD(확률적 경사 하강법)는 경사 하강법의 일종이고 model.parameters()를 이용하여 parameter를 전달함. lr은 학습률이며 momentum은 SGD에 관성을 더해줌(이전 이동 값을 고려하여 일정 비율만큼 다음 값을 결정)
model = Net()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# 훈련 과정
def train(epoch):
    # 학습 모드로 전환
    model.train()
    # train_loader로 학습
    # 각 data와 target을 전에 설정했던 device에 보냄
    # 미분을 통해 얻은 기울기가 이전에 계산된 기울기 값에 누적되기 때문에 기울기 값을 0으로 초기화 함
    # data를 model에 넣어 예측값 output을 도출
    # 손실값을 criterion함수를 이용해 도출 (예측값, 결과값)
    # 역전파 실행, 손실 함수를 미분하여 기울기 계산
    # step() 함수를 호출하여 parameter를 업데이트함
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        # 배치 사이즈가 10의 배수일때마다 학습을 반복한 수, Batch Status, loss값 출력
        if batch_idx % 10 == 0:
            print('Train Epoch: {} | Batch Status: {}/{} ({:.0f}%) | Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

# 테스트 과정
def test():
    # 평가 모드로 전환
    # 정확도 계산을 위한 변수 설정
    model.eval()
    test_loss = 0
    correct = 0
    # test_loader로 모델 시험
    # 각 data와 target을 전에 설정했던 device에 보냄
    # data를 model에 넣어 예측값 output을 도출
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        # sum up batch loss
        # 배치 손실값의 합
        test_loss += criterion(output, target).item()
        # get the index of the max
        # 최대값의 index를 반환/keepdim은 벡터 차원을 유지 시킬건지 아닌지를 설정
        # pred값과 target값을 비교하여 일치하는지 검사한 후 일치하는 것들의 개수의 합을 저장
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    # 테스트 손실값을 test_loader.dataset길이로 나눔
    # 테스트에 대한 손실값과 정확도를 계산
    test_loss /= len(test_loader.dataset)
    print(f'===========================\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.0f}%)')

# main에서 실행
# time 함수로 현재 시간 구함
# 9번 반복
if __name__ == '__main__':
    since = time.time()
    for epoch in range(1, 10):
        # 시작시간 구함
        # 학습 훈련을 시작함
        # 학습한 시간을 분과 초로 나눔
        # 학습한 시간 출력
        # 테스트를 시작함
        # 테스트한 시간을 분과 초로 나눔
        # 테스트한 시간 출력
        epoch_start = time.time()
        train(epoch)
        m, s = divmod(time.time() - epoch_start, 60)
        print(f'Training time: {m:.0f}m {s:.0f}s')
        test()
        m, s = divmod(time.time() - epoch_start, 60)
        print(f'Testing time: {m:.0f}m {s:.0f}s')

# 학습하고 테스한 시간을 분과 초로 나누고 출력
    m, s = divmod(time.time() - since, 60)
    print(f'Total Time: {m:.0f}m {s:.0f}s\nModel was trained on {device}!')
