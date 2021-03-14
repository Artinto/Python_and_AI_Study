from __future__ import print_function # Python3 에서 쓰던 문법을 Python 2 에서도 쓰게 해주는 Library
from torch import nn, optim, cuda
from torch.utils import data
from torchvision import datasets, transforms # torchvision : Package consists of popular datasets, model architectures
import torch.nn.functional as F
import time

# Training settings
batch_size = 64 # Epoch 에서 한 번에 학습하지 않고 나누어 학습할 경우의 그 사이즈 

device = 'cuda' if cuda.is_available() else 'cpu' # 사용가능하다면 연산에 GPU 사용 else CPU 사용 
print(f'Training MNIST Model on {device}\n{"=" * 44}')

# MNIST Dataset
train_dataset = datasets.MNIST(root='./mnist_data/', # File 경로
                               train=True,
                               transform=transforms.ToTensor(), # data 형태를 tensor type으로 형 변환
                               download=True) # dataset download 

test_dataset = datasets.MNIST(root='./mnist_data/',
                              train=False,
                              transform=transforms.ToTensor())

# Data Loader (Input Pipeline) 
train_loader = data.DataLoader(dataset=train_dataset,  # train_dataset을 batch size 만큼 불러들여 학습을 진행하게 해주는 dataloader
                                           batch_size=batch_size,
                                           shuffle=True) # data 섞기 Yes

test_loader = data.DataLoader(dataset=test_dataset,  # test_dataset을 batch size 만큼 불러들여 학습을 진행하게 해주는 dataloader
                                          batch_size=batch_size,
                                          shuffle=False) # data 섞기 No


class Net(nn.Module):

    def __init__(self): # Wide and deep  Input data (28x28 = 784) / Output data 10  # Hidden Layer = 3 layers
        super(Net, self).__init__()
        self.l1 = nn.Linear(784, 520)
        self.l2 = nn.Linear(520, 320)
        self.l3 = nn.Linear(320, 240)
        self.l4 = nn.Linear(240, 120)
        self.l5 = nn.Linear(120, 10)

    def forward(self, x):
        
        # Pytorch / Tensorflow 의 view 함수 : 원소의 수를 유지하면서 Tensor의 크기 변경 ==> Numpy의 reshape와 동일 
        
        # Flatten the data (n, 1, 28, 28)-> (n, 784) : 28x28의 1 Channel data n개 (4차원) -> 784 크기의 data n개 (2차원)
        x = x.view(-1, 784)  
        
        # Activation function -> relu function 
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        
        # 마지막 Layer는 Activation function으로  Softmax 함수를 통과시킬 것이기 때문에 Activation function을 적용하지 않고 return 
        return self.l5(x)


model = Net() # Net class 객체 생성
model.to(device) # if GPU 사용한다면 GPU에 객체 올리기 
criterion = nn.CrossEntropyLoss() # CrossEntropy 선언 
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5) # 추가적으로 momentum 파라미터를 활용하여 


def train(epoch):
    model.train()  # Net class 의 내장함수 train 
    for batch_idx, (data, target) in enumerate(train_loader): # train loader = (data , target) 
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


def test():
    model.eval()  # Net class 의 내장함수 evaluate  
    test_loss = 0 # Average Loss를 위해 선언
    correct = 0   # Accuracy 를 위해 선언
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        # sum up batch loss  = reduction = 'sum' 과 동일  
        test_loss += criterion(output, target).item() # batch 마다 나온 loss를 모두 합산 
        # get the index of the max
        pred = output.data.max(1, keepdim=True)[1] # 가장 큰 output의 index return (one-hot과 비슷)
        correct += pred.eq(target.data.view_as(pred)).cpu().sum() # target과 pred를 비교 

    test_loss /= len(test_loader.dataset) # Average Loss 
    print(f'===========================\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.0f}%)')


if __name__ == '__main__':
    
    # train과 test에 소요되는 시간 계산하는 Main 문
    since = time.time() # since : 시작 시간 
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
