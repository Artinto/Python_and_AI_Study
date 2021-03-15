# https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import print_function
# python 3에서 쓰던 문법을 2에서 쓸수 있게 해주는 문법. (f-string 때문인 듯)
from torch import nn, optim, cuda
from torch.utils import data # DataLoader, Dataset
from torchvision import datasets, transforms # MNIST, transform
import torch.nn.functional as F
import time

# Training settings
batch_size = 64
device = 'cuda' if cuda.is_available() else 'cpu' # GPU 사용가능하면 사용, 없으면 CPU
print(f'Training MNIST Model on {device}\n{"=" * 44}') # 어떤 device 사용하고 있는지 프린트
# f-string(포맷팅) 사용

# torchvision에서 제공하는 MNIST Dataset 사용
# './mnist_data/'에 없으면 download=True 다운로드하라. + 데이터들은 tensor로 바꾼다.
train_dataset = datasets.MNIST(root='./mnist_data/',train=True,transform=transforms.ToTensor(),download=True)

#testset은 train=False
test_dataset = datasets.MNIST(root='./mnist_data/', train=False,transform=transforms.ToTensor())
#test_dataset은 root='./mnist_data/'에 있는걸 사용한다.
 #이미 train_dataset에서 다운받음.


# Data Loader사용하여 batch_size만큼 데이터가져오기
train_loader = data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)

test_loader = data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(784, 520) #input : 784 (MNIST image : 28x28)
        self.l2 = nn.Linear(520, 320)
        self.l3 = nn.Linear(320, 240)
        self.l4 = nn.Linear(240, 120)
        self.l5 = nn.Linear(120, 10) #output : 10 (0~9)
        # layer 5개 쌓아 deep하게 만듦.

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten the data (n, 1, 28, 28)-> (n, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)


model = Net()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

 # 학습
def train(epoch):
    model.train()
    # 모델을 학습 모드로 변환 
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} | Batch Status: {}/{} ({:.0f}%) | Loss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),100. * batch_idx / len(train_loader), loss.item()))


def test():
    model.eval()
    # 모델을 평가 모드로 변환 
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        # sum up batch loss
        test_loss += criterion(output, target).item()
        # get the index of the max
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    
    test_loss /= len(test_loader.dataset)
    print(f'===========================\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} 'f'({100. * correct / len(test_loader.dataset):.0f}%)')


if __name__ == '__main__':
    since = time.time()
    for epoch in range(1, 10):
        epoch_start = time.time() # 타이머 : 초.마이크로초
        train(epoch)
        m, s = divmod(time.time() - epoch_start, 60) #경과시간을 분과 초로 나타내려고
        print(f'Training time: {m:.0f}m {s:.0f}s')
        test()
        m, s = divmod(time.time() - epoch_start, 60)
        print(f'Testing time: {m:.0f}m {s:.0f}s')

    m, s = divmod(time.time() - since, 60) # 에폭 다 돌았을 때 전체 걸린 시간
    print(f'Total Time: {m:.0f}m {s:.0f}s\nModel was trained on {device}!')
