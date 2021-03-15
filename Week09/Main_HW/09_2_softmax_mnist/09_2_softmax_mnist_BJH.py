# https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import print_function
from torch import nn, optim, cuda
from torch.utils import data
from torchvision import datasets, transforms
import torch.nn.functional as F
import time

# Training settings
batch_size = 64
device = 'cuda' if cuda.is_available() else 'cpu' #cuda는 엔디비아사의 GPU기술 AMD는 OPENCL만 가능...ㅠ
print(f'Training MNIST Model on {device}\n{"=" * 44}')

# MNIST Dataset
#Train과 Test에 차이를 둔다.
train_dataset = datasets.MNIST(root='./mnist_data/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='./mnist_data/',
                              train=False,
                              transform=transforms.ToTensor())

# Data Loader (Input Pipeline)

train_loader = data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__() #784개의 픽셀들을 신경망을 통하여 최종 10개로 만들기
        self.l1 = nn.Linear(784, 520)
        self.l2 = nn.Linear(520, 320)
        self.l3 = nn.Linear(320, 240)
        self.l4 = nn.Linear(240, 120)
        self.l5 = nn.Linear(120, 10)

    def forward(self, x): #위에서 정의했던 과정을 사용하는 것
        x = x.view(-1, 784)  # Flatten the data (n, 1, 28, 28)-> (n, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)  #return이기에 따로 함수로 안해준다.


model = Net()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5) #모멘텀은 전의 기울기를 약간 반영하는 식으로 계산


def train(epoch): #train
    model.train() #훈련할 때 불러오기
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} | Batch Status: {}/{} ({:.0f}%) | Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(): #test
    model.eval() #테스트 할 때 불러오기
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        # sum up batch loss
        test_loss += criterion(output, target).item()
        # get the index of the max
        pred = output.data.max(1, keepdim=True)[1] #인덱스 출력
        correct += pred.eq(target.data.view_as(pred)).cpu().sum() #correct 와 pred.eq를 비교하여 맞으면 계속 더함

    test_loss /= len(test_loader.dataset) #갯수만큼 나눠주기
    print(f'===========================\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.0f}%)') #출력


if __name__ == '__main__': #메인문
    since = time.time()
    for epoch in range(1, 10):
        epoch_start = time.time() #현재시각을 계산해주나 반환초기에 조금의 과정을 거쳐야 사람이 직관적으로 확인가능
        train(epoch)
        m, s = divmod(time.time() - epoch_start, 60) #경과 시간 계산하기
        print(f'Training time: {m:.0f}m {s:.0f}s')
        test()
        m, s = divmod(time.time() - epoch_start, 60) #경과 시간 계산하기
        print(f'Testing time: {m:.0f}m {s:.0f}s')

    m, s = divmod(time.time() - since, 60)
    print(f'Total Time: {m:.0f}m {s:.0f}s\nModel was trained on {device}!')