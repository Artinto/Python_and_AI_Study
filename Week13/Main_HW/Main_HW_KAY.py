import torch
import numpy as np
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
#torchvision에서 제공하는 imagefodler를 사용 randomcrop를 이용하여 이미지를 자르고 
#텐서로 변환 정규화를 통하여 rbg컬러를 0.5로 맞추어 주었다
trainset = ImageFolder("/content/drive/MyDrive/train",transform=transforms.Compose([transforms.RandomCrop(100),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]))

testset = ImageFolder("/content/drive/MyDrive/test",transform=transforms.Compose([transforms.RandomCrop(100), transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]))
                                                      
train_loader = data.DataLoader(trainset, batch_size=64, shuffle=True)
test_loader = data.DataLoader(testset, batch_size=64, shuffle=True)


class InceptionA(nn.Module):

    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=1) #1x1 아웃풋 16, kernel_size:window size

        self.branch5x5_1 = nn.Conv2d(in_channels, 16, kernel_size=1) #1x1 아웃풋 16,
        self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2) #5x5필터링,padding:주변을 숫자로 둘러주는 것(같은 사이즈 유지를 위해),  input이 16, output이 24

        self.branch3x3dbl_1 = nn.Conv2d(in_channels, 16, kernel_size=1) #1x1 아웃풋 16,
        self.branch3x3dbl_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1) #input이 16, output이 24, 3*3
        self.branch3x3dbl_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)# output이 24, 3*3

        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)
 #conv2d:filter로 특징을 뽑아내는 레이어
    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1) #3*3필터, stride=1이므로 한칸식 이동
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool] 
        return torch.cat(outputs, 1) #텐서들을 연결 dim=1은 두번째 차원을 늘리라는 것 즉


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5) #합성곱 연산, input=1,outpiut=10, 필터=5*5 
        self.conv2 = nn.Conv2d(88, 20, kernel_size=5) #합성곱 연산,input=88, ouput=20,   필터=5*5, 16+24+24+24

        self.incept1 = InceptionA(in_channels=10)
        self.incept2 = InceptionA(in_channels=20)

        self.mp = nn.MaxPool2d(2) #2*2사이즈의 필터사용, 최댓값을 뽑아낸다, stride=1
        self.fc = nn.Linear(42592, 10)

    def forward(self, x):
        in_size = x.size(0)#(a, b)중 a를 받음
        x = F.relu(self.mp(self.conv1(x)))
        x = self.incept1(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = self.incept2(x)
        x = x.view(in_size, -1)  # 일자로 핌
        x = self.fc(x)
        return F.log_softmax(x, dim=0)#dim=0 열의 크기가 하나


model = Net()

optimizer = optim.SGD(model.parameters(), lr=0.015, momentum=0.6)


def train(epoch):
    model.train() #학습
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target) #data를 입력과 결과값으로 분리
        optimizer.zero_grad()#미분값 초기화
        output = model(data) #예상값
        loss = F.nll_loss(output, target) #loss계산, softmax는 이미 해줌
        loss.backward()#역전파 실행  
        optimizer.step()# 파라미터 이동
        if batch_idx % 10 == 0: #loss값 
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.data))


def test():
    model.eval() #eval 모드에서 사용
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target) #data를 입력과 결과값
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).data
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1] #예측값 output의 최댓값을 pred에
        correct += pred.eq(target.data.view_as(pred)).cpu().sum() # 맞은갯수 계산

    test_loss /= len(test_loader.dataset)#accuracy 출력
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))


for epoch in range(1, 10):
    train(epoch)
    test()
    
    
    Train Epoch: 1 [0/1949 (0%)]	Loss: 4.159012
Train Epoch: 1 [640/1949 (32%)]	Loss: 4.140414
Train Epoch: 1 [1280/1949 (65%)]	Loss: 4.089277
Train Epoch: 1 [870/1949 (97%)]	Loss: 3.313772
/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:102: UserWarning: volatile was removed and now has no effect. Use `with torch.no_grad():` instead.
/usr/local/lib/python3.7/dist-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='sum' instead.
  warnings.warn(warning.format(ret))

Test set: Average loss: 3.8415, Accuracy: 39/100 (39%)

Train Epoch: 2 [0/1949 (0%)]	Loss: 4.087945
Train Epoch: 2 [640/1949 (32%)]	Loss: 4.069433
Train Epoch: 2 [1280/1949 (65%)]	Loss: 4.045985
Train Epoch: 2 [870/1949 (97%)]	Loss: 3.227018

Test set: Average loss: 3.8503, Accuracy: 40/100 (40%)

Train Epoch: 3 [0/1949 (0%)]	Loss: 4.030289
Train Epoch: 3 [640/1949 (32%)]	Loss: 4.109765
Train Epoch: 3 [1280/1949 (65%)]	Loss: 3.957823
Train Epoch: 3 [870/1949 (97%)]	Loss: 3.298219

Test set: Average loss: 3.8182, Accuracy: 48/100 (48%)

Train Epoch: 4 [0/1949 (0%)]	Loss: 4.046031
Train Epoch: 4 [640/1949 (32%)]	Loss: 4.010128
Train Epoch: 4 [1280/1949 (65%)]	Loss: 4.074433
Train Epoch: 4 [870/1949 (97%)]	Loss: 3.218896

Test set: Average loss: 4.0300, Accuracy: 31/100 (31%)

Train Epoch: 5 [0/1949 (0%)]	Loss: 4.190258
Train Epoch: 5 [640/1949 (32%)]	Loss: 4.085792
Train Epoch: 5 [1280/1949 (65%)]	Loss: 4.037814
Train Epoch: 5 [870/1949 (97%)]	Loss: 3.234454

Test set: Average loss: 3.8067, Accuracy: 42/100 (42%)

Train Epoch: 6 [0/1949 (0%)]	Loss: 3.928722
Train Epoch: 6 [640/1949 (32%)]	Loss: 3.993922
Train Epoch: 6 [1280/1949 (65%)]	Loss: 3.872353
Train Epoch: 6 [870/1949 (97%)]	Loss: 3.297552

Test set: Average loss: 3.8588, Accuracy: 35/100 (35%)

Train Epoch: 7 [0/1949 (0%)]	Loss: 4.061032
Train Epoch: 7 [640/1949 (32%)]	Loss: 3.932746
Train Epoch: 7 [1280/1949 (65%)]	Loss: 4.024937
Train Epoch: 7 [870/1949 (97%)]	Loss: 3.433298

Test set: Average loss: 3.7879, Accuracy: 39/100 (39%)

Train Epoch: 8 [0/1949 (0%)]	Loss: 3.983347
Train Epoch: 8 [640/1949 (32%)]	Loss: 4.041204
Train Epoch: 8 [1280/1949 (65%)]	Loss: 3.965430
Train Epoch: 8 [870/1949 (97%)]	Loss: 3.249194

Test set: Average loss: 3.7067, Accuracy: 56/100 (56%)

Train Epoch: 9 [0/1949 (0%)]	Loss: 3.969170
Train Epoch: 9 [640/1949 (32%)]	Loss: 3.973055
Train Epoch: 9 [1280/1949 (65%)]	Loss: 4.059671
Train Epoch: 9 [870/1949 (97%)]	Loss: 3.115010

Test set: Average loss: 3.6388, Accuracy: 53/100 (53%)
