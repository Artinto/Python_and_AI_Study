from __future__ import print_function
from torch import nn, optim, cuda
from torch.utils import data
from torchvision import datasets, transforms
import torch.nn.functional as F
import time

# Training settings
batch_size = 64
device = 'cuda' if cuda.is_available() else 'cpu' #cuda(gpu)가 사용가능하면 쓰고 안되면 cpu
print(f'Training MNIST Model on {device}\n{"=" * 44}')

# MNIST Dataset
train_dataset = datasets.MNIST(root='/mnist_data/',   #학습데이터 설정
                               train=True,
                               transform=transforms.ToTensor(), #일반 이미지를 pytorch.tensor로 변환
                               download=True)

test_dataset = datasets.MNIST(root='./mnist_data/', #테스트 데이터 설정
                              train=False,
                              transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = data.DataLoader(dataset=train_dataset,  #트레인 로더에 학습 데이터 설정
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = data.DataLoader(dataset=test_dataset, #테스트 로더에 시험 데이터 설정
                                          batch_size=batch_size,
                                          shuffle=False)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()  #784개의 픽셀인 minist데이터를 10개의 아웃풋 레이블로 변환
        self.l1 = nn.Linear(784, 520) #히든 레이어를 추가해서 아웃풋을 계산한다
        self.l2 = nn.Linear(520, 320)
        self.l3 = nn.Linear(320, 240)
        self.l4 = nn.Linear(240, 120)
        self.l5 = nn.Linear(120, 10)

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten the data (n, 1, 28, 28)-> (n, 784)
        x = F.relu(self.l1(x)) #relu를 이용해서 데이터x를 linear모듈에 넣어준다
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x) #맨마지막 층은 softmax에서 logit으로 활용되기때문에 계산하지 않는다


model = Net()
model.to(device) #to(device)는 gpu를 사용할 경우
criterion = nn.CrossEntropyLoss() #CELoss 사용
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch): #학습 진행
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader): #트레인 로더의 데이터로 학습진행
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data) #예측값
        loss = criterion(output, target) #loss값 계산
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} | Batch Status: {}/{} ({:.0f}%) | Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item())) #정확도 출력

def test(): #테스트 진행
    model.eval() #model을 evaluation mode로 설정
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        # sum up batch loss
        test_loss += criterion(output, target).item() #loss값의 합
        # get the index of the max
        pred = output.data.max(1, keepdim=True)[1] #배열의 최대 값이 들어있는 index를 리턴
        correct += pred.eq(target.data.view_as(pred)).cpu().sum() #pred배열과 data의 일치를 검사하고 합을 구함

    test_loss /= len(test_loader.dataset)
    print(f'===========================\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.0f}%)') #테스트의 정확도 출력

if __name__ == '__main__': #메인에서 실행
    since = time.time()
    for epoch in range(1, 10): 
        epoch_start = time.time()
        train(epoch)
        m, s = divmod(time.time() - epoch_start, 60) #몫과 나머지를 m,s에 저장
        print(f'Training time: {m:.0f}m {s:.0f}s') #학습 속도 출력
        test()
        m, s = divmod(time.time() - epoch_start, 60)
        print(f'Testing time: {m:.0f}m {s:.0f}s')

    m, s = divmod(time.time() - since, 60)
    print(f'Total Time: {m:.0f}m {s:.0f}s\nModel was trained on {device}!')
