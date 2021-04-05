from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# Training settings
batch_size = 64


# MNIST Dataset 불러오기
# Train set , Test set 분리

train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

# Data Loader (Input Pipeline) -> Batch 학습 
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


class InceptionA(nn.Module):

    # Inception은 여러 종류의 filter를 병렬적으로(graph 상의 개념) 통과하게 한 후 그 결과들을 list 형태로 붙여 다음 layer로 넘겨준다.
    def __init__(self, in_channels): 
        # in_chnnels : 입력 채널 ( ex) grayscale = 1 channel, RGG scale = 3 channel ... )
        super(InceptionA, self).__init__()
        
        # 이렇게 사용하면 최종적인 출력 채널과 사이즈는 같지만 연산 과정에서 필요한 컴퓨팅 파워가 굉장히 줄어든다
        # 총 네 개의 Block에 대한 연산을 수행한 후 하나의 List로 만들어 다음 layer로 넘겨 줄 것
        # 입력 channel = in_channel , 출력 channel = 24
        # 첫 번째 Block
        self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=1) # 입력 채널 수 x 1 x 1 의 filter (bottle neck)   

        # 두 번째 Block
        self.branch5x5_1 = nn.Conv2d(in_channels, 16, kernel_size=1) # 입력 채널 수 x 1 x 1 
        self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2) # 16x5x5 subconvolution.  입력 채널 : 16, 출력 채널 : 24, padding size = 2 

        # 세 번째 Block
        self.branch3x3dbl_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3dbl_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)

        # 네 번째 Bloc
        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        # 네 개의 Block에 대한 출력 결과를 하나의 list로 
        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        
        # torch.cat : tensor를 이어붙이는 함수 cat(Tensors, dimension)
        return torch.cat(outputs, 1) # outputs tensor들을 dim = 1로 이어 붙인다.
        # ex) A = tensor([[1,2,3],[4,5,6]]) , B = tensor([[7,8,9],[10,11,12]])
        #  C = torch.cat(A, dim = 1)  -> C = tensor([[1,2,3,7,8,9],[4,5,6,10,11,12]])

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(88, 20, kernel_size=5)

        self.incept1 = InceptionA(in_channels=10)
        self.incept2 = InceptionA(in_channels=20)

        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(1408, 10) # flattend data 넣어주는 fully-connected layer 

    def forward(self, x):
        in_size = x.size(0)
        
        # data in -> convolution -> maxpooling -> relu activation function
        x = F.relu(self.mp(self.conv1(x)))
        # relu activation function -> inception layer (4 개의 block에 대한 연산을 통과한 후 list 형태로 합하여 다시 return)
        x = self.incept1(x)
        # inception layer1 -> convolution -> maxpooling -> relu activation function
        x = F.relu(self.mp(self.conv2(x)))
        # relu activation function -> inception layer
        x = self.incept2(x)
        # inception layer2 -> flatten data 
        x = x.view(in_size, -1)  # flatten the tensor 
        # flatten data to fully connected layer 
        x = self.fc(x)
        # 마지막으로 loss function 에 값 전달 
        return F.log_softmax(x)


model = Net()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target) # log_softmax를 사용했으므로 nll_loss를 통해 loss 계산 
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).data[0]
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, 10):
    train(epoch)
    test()
