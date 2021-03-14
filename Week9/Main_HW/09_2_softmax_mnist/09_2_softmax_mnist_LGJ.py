# https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import print_function # __future__에서 print_function을 불러옴
from torch import nn, optim, cuda # torch에서 nn, optim, cuda를 불러옴
from torch.utils import data  # torch.utils에서 data를 불러옴
from torchvision import datasets, transforms  # torchvision에서 datasets, transforms를 불러옴
import torch.nn.functional as F # torch.nn.functional을 F라 선언
import time # time 불러옴

# Training settings
batch_size = 64 # 배치 사이즈 64
device = 'cuda' if cuda.is_available() else 'cpu' # 가능하다면 'cuda'(GPU)가속을 사용
print(f'Training MNIST Model on {device}\n{"=" * 44}') 

# MNIST Dataset
train_dataset = datasets.MNIST(root='./mnist_data/',  # datasets를 통하여 MNIST데이터를 불러옴
                               train=True,  # 훈련 데이터를 반환받음
                               transform=transforms.ToTensor(), # 현재 데이터를 파이토치 텐서로 변환
                               download=True) # 해당 경로에 MNIST데이터가 없다면 다운로드 받음
  
test_dataset = datasets.MNIST(root='./mnist_data/',  # datasets를 통하여 MNIST데이터를 불러옴
                              train=False,  # 테스트 데이터를 반환받음
                              transform=transforms.ToTensor()) # 현재 데이터를 파이토치 텐서로 변환

# Data Loader (Input Pipeline)
train_loader = data.DataLoader(dataset=train_dataset, # dataloader를 사용하여 train_dataset를 불러오고, 64의 배치사이즈 지정 및 셔플 활성화
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = data.DataLoader(dataset=test_dataset, # dataloader를 사용하여 train_dataset를 불러오고, 64의 배치사이즈 지정 및 셔플 비활성화
                                          batch_size=batch_size,
                                          shuffle=False)


class Net(nn.Module): # nn.Module의 상속을 받는 신경망 클래스 작성

    def __init__(self): 
        super(Net, self).__init__() 
        self.l1 = nn.Linear(784, 520) # input layer
        self.l2 = nn.Linear(520, 320) # hidden layer1
        self.l3 = nn.Linear(320, 240) # hidden layer2
        self.l4 = nn.Linear(240, 120) # hidden layer3
        self.l5 = nn.Linear(120, 10) # output layer

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten the data (n, 1, 28, 28)-> (n, 784)
        x = F.relu(self.l1(x))  # 평탄화된 x를 선형함수 l1에 넣고, 활성화 relu함수에 또 넣은 뒤 결과를 x에 덮어씌움
        x = F.relu(self.l2(x))  # 평탄화된 x를 선형함수 l2에 넣고, 활성화 relu함수에 또 넣은 뒤 결과를 x에 덮어씌움
        x = F.relu(self.l3(x))  # 평탄화된 x를 선형함수 l3에 넣고, 활성화 relu함수에 또 넣은 뒤 결과를 x에 덮어씌움
        x = F.relu(self.l4(x))  # 평탄화된 x를 선형함수 l4에 넣고, 활성화 relu함수에 또 넣은 뒤 결과를 x에 덮어씌움
        return self.l5(x) # l5를 출력


model = Net() # model에Net 클래스 적용
model.to(device)  # GPU가속 사용
criterion = nn.CrossEntropyLoss() # criterion에 nn.CrossEntropyLoss()할당. 
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)  # 최적화 함수로 경사하강법을 실행


def train(epoch): #
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader): # train_loader를 이용하여 각각의 배치사이즈의 data, target를 불러옴
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() # 미분값 초기화
        output = model(data)  
        loss = criterion(output, target)  # 예측값 output과 target의 손실 계산
        loss.backward() # 역전파 실행
        optimizer.step()  # 변수 갱신
        if batch_idx % 10 == 0:
            print('Train Epoch: {} | Batch Status: {}/{} ({:.0f}%) | Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(): # 학습이 완료된 모델에 test데이터셋을 적용하여 제대로 학습이 되었나 확인하는 과정
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:  # test_loader
        data, target = data.to(device), target.to(device) # 연산에서 GPU가속 사용
        output = model(data)  # 모델에 데이터를 주입
        # sum up batch loss
        test_loss += criterion(output, target).item() # 테스트 손실 값을 계산
        # get the index of the max
        pred = output.data.max(1, keepdim=True)[1]  # output.data.max함수로 예측
        correct += pred.eq(target.data.view_as(pred)).cpu().sum() # 예측값과 타겟 데이터를 비교하여 얼마나 옳았는지 합을 계산(cpu로 연산)

    test_loss /= len(test_loader.dataset)  
    print(f'===========================\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.0f}%)')  # 예측이 얼마나 맞았는지 


if __name__ == '__main__':
    since = time.time()
    for epoch in range(1, 10):
        epoch_start = time.time()
        train(epoch)
        m, s = divmod(time.time() - epoch_start, 60)
        print(f'Training time: {m:.0f}m {s:.0f}s')
        test()
        m, s = divmod(time.time() - epoch_start, 60)
        print(f'Testing time: {m:.0f}m {s:.0f}s')

    m, s = divmod(time.time() - since, 60)
    print(f'Total Time: {m:.0f}m {s:.0f}s\nModel was trained on {device}!')
