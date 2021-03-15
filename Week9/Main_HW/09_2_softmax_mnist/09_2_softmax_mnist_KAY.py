# https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import print_function
from torch import nn, optim, cuda
from torch.utils import data
from torchvision import datasets, transforms
import torch.nn.functional as F
import time

# Training settings
batch_size = 64#배치사이즈 설정
device = 'cuda' if cuda.is_available() else 'cpu'#cuda가 사용가능하면 cuda 아님 cpu
print(f'Training MNIST Model on {device}\n{"=" * 44}')

# MNIST Dataset
train_dataset = datasets.MNIST(root='./mnist_data/',#학습데이터 설정
                               train=True,
                               transform=transforms.ToTensor(),#이미지를 tensor로 변환
                               download=True)

test_dataset = datasets.MNIST(root='./mnist_data/',#테스트 데이터
                              train=False,
                              transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = data.DataLoader(dataset=train_dataset,#트레인 로더에 학습데이터 설정
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = data.DataLoader(dataset=test_dataset,#테스트 데이터 설정
                                          batch_size=batch_size,
                                          shuffle=False)


class Net(nn.Module):

    def __init__(self):#784개의 데이터를 10개로 바꿈
        super(Net, self).__init__()
        self.l1 = nn.Linear(784, 520)
        self.l2 = nn.Linear(520, 320)
        self.l3 = nn.Linear(320, 240)
        self.l4 = nn.Linear(240, 120)
        self.l5 = nn.Linear(120, 10)

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten the data (n, 1, 28, 28)-> (n, 784)
        x = F.relu(self.l1(x))#활성화함수 relu를 사용하여 0이하의 수는 0으로 반환
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)#마지막은 softmax에서 사용되기 때문에 계산 않함


model = Net()
model.to(device)#gpu사용시
criterion = nn.CrossEntropyLoss()#celoss
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):#학습
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):#트레인노더 데이터 이용
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)#예측값
        loss = criterion(output, target)#loss
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} | Batch Status: {}/{} ({:.0f}%) | Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))#정확도


def test():#테스트
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        # sum up batch loss
        test_loss += criterion(output, target).item()#loss의 합
        # get the index of the max
        pred = output.data.max(1, keepdim=True)[1]#배열의 최대값 
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()#pred와 data가 같은지 확인하고 더함

    test_loss /= len(test_loader.dataset)
    print(f'===========================\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.0f}%)')#정황도


if __name__ == '__main__':#메인
    since = time.time()
    for epoch in range(1, 10):
        epoch_start = time.time()
        train(epoch)
        m, s = divmod(time.time() - epoch_start, 60)#몫과 나머지 저장
        print(f'Training time: {m:.0f}m {s:.0f}s')#
        test()
        m, s = divmod(time.time() - epoch_start, 60)
        print(f'Testing time: {m:.0f}m {s:.0f}s')

    m, s = divmod(time.time() - since, 60)
    print(f'Total Time: {m:.0f}m {s:.0f}s\nModel was trained on {device}!')
