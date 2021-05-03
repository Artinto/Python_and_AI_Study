import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable

tf = transforms.Compose([transforms.Resize((100,100)), transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])

modelset = torchvision.datasets.ImageFolder(root = '/content/drive/MyDrive/origin_data',transform=tf)

trainset, testset = torch.utils.data.random_split(modelset,[400, 151])

train_loader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=True, num_workers=0)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # Conv2d(input channel, output channel, kernel size/filter size)
        # Convolutional Layer. In channel = 1, Out channel = 10, filter의 사이즈는 5x5
        # Default value => padding = 0 , stride = 1 
        
        # 입력 크기 : 1 x 100 x 100
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)   
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # 출력 Channel = 20
        
        self.mp = nn.MaxPool2d(2) 
        # Max Pooling kernel size = 2x2
        # Default value => padding = 0 , stride = 1 
        
        self.fc = nn.Linear(9680, 2) # 공간 데이터를 Flatten -> Dense Net 

    def forward(self, x):
        in_size = x.size(0) # 들어온 data의 개수 -> data.shape = (n, 1, 100, 100) -> n = batch size 
        x = F.relu(self.mp(self.conv1(x))) # Convolution 수행 -> Maxpool -> relu activation function 1
        x = F.relu(self.mp(self.conv2(x))) # Convolution 수행 후 Maxpool -> relu activation function 2
        x = x.view(in_size, -1)  # flatten the tensor
        x = self.fc(x) # data를 flatten 한 후 Dense Net에 넣어줌 
        return F.log_softmax(x) # Calc Loss


model = Net()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
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
    for data, target in test_loader:
        # volatile -> 변수를 CPU Register가 아니라 Memory에 저장한다.
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).data
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1] # One - Hot 과 유사한 개념 . 가장 큰 Class의 Index
        correct += pred.eq(target.data.view_as(pred)).cpu().sum() # Target 과 pred 비교 후 맞은 개수 누적합

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, 10):
    train(epoch)
    test()


"""
Train Epoch: 1 [0/400 (0%)]	Loss: 0.699748
Train Epoch: 1 [160/400 (40%)]	Loss: 0.634114
Train Epoch: 1 [320/400 (80%)]	Loss: 0.088880
/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:80: UserWarning: volatile was removed and now has no effect. Use `with torch.no_grad():` instead.
/usr/local/lib/python3.7/dist-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='sum' instead.
  warnings.warn(warning.format(ret))

Test set: Average loss: 0.0358, Accuracy: 151/151 (100%)

Train Epoch: 2 [0/400 (0%)]	Loss: 0.017059
Train Epoch: 2 [160/400 (40%)]	Loss: 0.012337
Train Epoch: 2 [320/400 (80%)]	Loss: 0.010243

Test set: Average loss: 0.0071, Accuracy: 151/151 (100%)

Train Epoch: 3 [0/400 (0%)]	Loss: 0.003861
Train Epoch: 3 [160/400 (40%)]	Loss: 0.003829
Train Epoch: 3 [320/400 (80%)]	Loss: 0.001962

Test set: Average loss: 0.0036, Accuracy: 151/151 (100%)

Train Epoch: 4 [0/400 (0%)]	Loss: 0.004156
Train Epoch: 4 [160/400 (40%)]	Loss: 0.001108
Train Epoch: 4 [320/400 (80%)]	Loss: 0.002506

Test set: Average loss: 0.0024, Accuracy: 151/151 (100%)

Train Epoch: 5 [0/400 (0%)]	Loss: 0.000958
Train Epoch: 5 [160/400 (40%)]	Loss: 0.000886
Train Epoch: 5 [320/400 (80%)]	Loss: 0.000449

Test set: Average loss: 0.0017, Accuracy: 151/151 (100%)

Train Epoch: 6 [0/400 (0%)]	Loss: 0.000940
Train Epoch: 6 [160/400 (40%)]	Loss: 0.001652
Train Epoch: 6 [320/400 (80%)]	Loss: 0.001075

Test set: Average loss: 0.0014, Accuracy: 151/151 (100%)

Train Epoch: 7 [0/400 (0%)]	Loss: 0.002108
Train Epoch: 7 [160/400 (40%)]	Loss: 0.003509
Train Epoch: 7 [320/400 (80%)]	Loss: 0.000894

Test set: Average loss: 0.0011, Accuracy: 151/151 (100%)

Train Epoch: 8 [0/400 (0%)]	Loss: 0.000993
Train Epoch: 8 [160/400 (40%)]	Loss: 0.000532
Train Epoch: 8 [320/400 (80%)]	Loss: 0.001039

Test set: Average loss: 0.0009, Accuracy: 151/151 (100%)

Train Epoch: 9 [0/400 (0%)]	Loss: 0.000811
Train Epoch: 9 [160/400 (40%)]	Loss: 0.000578
Train Epoch: 9 [320/400 (80%)]	Loss: 0.000482

Test set: Average loss: 0.0008, Accuracy: 151/151 (100%)
"""
