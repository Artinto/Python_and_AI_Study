# https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import print_function # __future__ : python3에서 python2 문법 사용 가능
from torch import nn, optim, cuda 
from torch.utils import data # for using DataLoader
# torchvision : 이미지 변환을 위해
# datasets : for using MNIST
# transforms : 다양한 이미지 변환 제공 (현재 과제에서는 numpy -> tensor)
from torchvision import datasets, transforms
import torch.nn.functional as F
import time

# Training settings
batch_size = 64 # batch size(64) 지정
device = 'cuda' if cuda.is_available() else 'cpu' # cuda 사용
print(f'Training MNIST Model on {device}\n{"=" * 44}')

# MNIST Dataset
train_dataset = datasets.MNIST(root='./mnist_data/', # data path
                               train=True, # train data로 사용할지 안할지
                               transform=transforms.ToTensor(), # 데이터 형식 지정
                               download=True) # MNIST 없으면 download

test_dataset = datasets.MNIST(root='./mnist_data/',
                              train=False,
                              transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True) # data shuffle

test_loader = data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(784, 520)
        self.l2 = nn.Linear(520, 320)
        self.l3 = nn.Linear(320, 240)
        self.l4 = nn.Linear(240, 120)
        self.l5 = nn.Linear(120, 10)  # **10차원의 output**

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten the data (n, 1, 28, 28)-> (n, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)


model = Net()
model.to(device)
criterion = nn.CrossEntropyLoss() # CrossEntropyLoss 사용
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5) # optimizer로 SGD사용, learning rate = 0.01
                                                                 # momentum을 사용해서 local minimum을 탈출 가능


def train(epoch):
    model.train() # (Trian mode) 학습할 땐 무작위로 노드를 선택하여 선별적으로 노드를 활용함
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() # 기울기 0으로 초기화
        output = model(data)  # 학습
        loss = criterion(output, target) # get loss
        loss.backward()  # 역전파
        optimizer.step() # 매개변수 갱신
        if batch_idx % 10 == 0:
            print('Train Epoch: {} | Batch Status: {}/{} ({:.0f}%) | Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test():
    model.eval() # (Evalutaion mode) 평가하는 과정에서는 모든 노드를 사용하겠다는 의미
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        # sum up batch loss
        test_loss += criterion(output, target).item()
        # get the index of the max
        pred = output.data.max(1, keepdim=True)[1] # One hot 벡터에서 값이 1인 index 가져오기
        correct += pred.eq(target.data.view_as(pred)).cpu().sum() # target과 pred를 비교하여 성공률 구하기

    test_loss /= len(test_loader.dataset)
    print(f'===========================\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.0f}%)')


if __name__ == '__main__':
    since = time.time() # 현재 시간
    for epoch in range(1, 10):
        epoch_start = time.time() # 시작 시간
        train(epoch) # start train
        m, s = divmod(time.time() - epoch_start, 60) # check train time
        print(f'Training time: {m:.0f}m {s:.0f}s')
        test() # start test
        m, s = divmod(time.time() - epoch_start, 60) # check test time
        print(f'Testing time: {m:.0f}m {s:.0f}s')

    m, s = divmod(time.time() - since, 60) # check total time
    print(f'Total Time: {m:.0f}m {s:.0f}s\nModel was trained on {device}!')