from PIL import Image
import numpy as np
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
from google.colab import drive

batch_size = 16

trans=transforms.Compose([transforms.Resize((100,100)),
                          transforms.ToTensor(),
                          transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

train_dataset = ImageFolder("/content/gdrive/My Drive/Week13/custom_dataset/train", transform=trans)
                               
test_dataset = ImageFolder("/content/gdrive/My Drive/Week13/custom_dataset/test", transform=trans)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers = 4)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=True, num_workers = 4)


class InceptionA(nn.Module):

    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)

        self.branch5x5_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3dbl_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)

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

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(88, 20, kernel_size=5)

        self.incept1 = InceptionA(in_channels=10)
        self.incept2 = InceptionA(in_channels=20)

        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(42592, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = self.incept1(x)
        x = F.relu(self.mp(self.conv2(x)))
from PIL import Image
import numpy as np
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
from google.colab import drive

batch_size = 16

trans=transforms.Compose([transforms.Resize((100,100)),
                          transforms.ToTensor(),
                          transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

train_dataset = ImageFolder("/content/gdrive/My Drive/Week13/custom_dataset/train", transform=trans)
                               
test_dataset = ImageFolder("/content/gdrive/My Drive/Week13/custom_dataset/test", transform=trans)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers = 4)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=True, num_workers = 4)


class InceptionA(nn.Module):

    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)

        self.branch5x5_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3dbl_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)

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

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(88, 20, kernel_size=5)

        self.incept1 = InceptionA(in_channels=10)
        self.incept2 = InceptionA(in_channels=20)

        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(42592, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = self.incept1(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = self.incept2(x)
        x = x.view(in_size, -1)
        x = self.fc(x)
        return F.log_softmax(x)


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
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).data
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

/usr/local/lib/python3.7/dist-packages/torch/utils/data/dataloader.py:477: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  cpuset_checked))
/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:82: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
Train Epoch: 1 [0/1949 (0%)]	Loss: 2.262018
Train Epoch: 1 [160/1949 (8%)]	Loss: 1.726339
Train Epoch: 1 [320/1949 (16%)]	Loss: 1.421450
Train Epoch: 1 [480/1949 (25%)]	Loss: 1.556488
Train Epoch: 1 [640/1949 (33%)]	Loss: 1.629135
Train Epoch: 1 [800/1949 (41%)]	Loss: 1.386070
Train Epoch: 1 [960/1949 (49%)]	Loss: 1.423238
Train Epoch: 1 [1120/1949 (57%)]	Loss: 1.668453
Train Epoch: 1 [1280/1949 (66%)]	Loss: 1.497475
Train Epoch: 1 [1440/1949 (74%)]	Loss: 1.521662
Train Epoch: 1 [1600/1949 (82%)]	Loss: 1.283526
Train Epoch: 1 [1760/1949 (90%)]	Loss: 1.391987
Train Epoch: 1 [1920/1949 (98%)]	Loss: 1.515293
/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:110: UserWarning: volatile was removed and now has no effect. Use `with torch.no_grad():` instead.
/usr/local/lib/python3.7/dist-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='sum' instead.
  warnings.warn(warning.format(ret))

Test set: Average loss: 1.1862, Accuracy: 43/100 (43%)

Train Epoch: 2 [0/1949 (0%)]	Loss: 1.041221
Train Epoch: 2 [160/1949 (8%)]	Loss: 1.250063
Train Epoch: 2 [320/1949 (16%)]	Loss: 0.733387
Train Epoch: 2 [480/1949 (25%)]	Loss: 1.581814
Train Epoch: 2 [640/1949 (33%)]	Loss: 1.026476
Train Epoch: 2 [800/1949 (41%)]	Loss: 1.495665
Train Epoch: 2 [960/1949 (49%)]	Loss: 0.725726
Train Epoch: 2 [1120/1949 (57%)]	Loss: 0.956531
Train Epoch: 2 [1280/1949 (66%)]	Loss: 0.744177
Train Epoch: 2 [1440/1949 (74%)]	Loss: 0.831996
Train Epoch: 2 [1600/1949 (82%)]	Loss: 0.739115
Train Epoch: 2 [1760/1949 (90%)]	Loss: 0.693679
Train Epoch: 2 [1920/1949 (98%)]	Loss: 0.663333

Test set: Average loss: 0.7141, Accuracy: 68/100 (68%)

Train Epoch: 3 [0/1949 (0%)]	Loss: 0.667219
Train Epoch: 3 [160/1949 (8%)]	Loss: 0.856315
Train Epoch: 3 [320/1949 (16%)]	Loss: 1.236073
Train Epoch: 3 [480/1949 (25%)]	Loss: 1.317776
Train Epoch: 3 [640/1949 (33%)]	Loss: 0.639973
Train Epoch: 3 [800/1949 (41%)]	Loss: 0.608828
Train Epoch: 3 [960/1949 (49%)]	Loss: 0.832086
Train Epoch: 3 [1120/1949 (57%)]	Loss: 0.666449
Train Epoch: 3 [1280/1949 (66%)]	Loss: 0.420417
Train Epoch: 3 [1440/1949 (74%)]	Loss: 1.146005
Train Epoch: 3 [1600/1949 (82%)]	Loss: 0.703503
Train Epoch: 3 [1760/1949 (90%)]	Loss: 0.922001
Train Epoch: 3 [1920/1949 (98%)]	Loss: 0.509392

Test set: Average loss: 1.1261, Accuracy: 56/100 (56%)

Train Epoch: 4 [0/1949 (0%)]	Loss: 0.642347
Train Epoch: 4 [160/1949 (8%)]	Loss: 0.563760
Train Epoch: 4 [320/1949 (16%)]	Loss: 0.520198
Train Epoch: 4 [480/1949 (25%)]	Loss: 0.482343
Train Epoch: 4 [640/1949 (33%)]	Loss: 0.606384
Train Epoch: 4 [800/1949 (41%)]	Loss: 0.586854
Train Epoch: 4 [960/1949 (49%)]	Loss: 0.480791
Train Epoch: 4 [1120/1949 (57%)]	Loss: 0.321699
Train Epoch: 4 [1280/1949 (66%)]	Loss: 0.850219
Train Epoch: 4 [1440/1949 (74%)]	Loss: 0.692008
Train Epoch: 4 [1600/1949 (82%)]	Loss: 0.452003
Train Epoch: 4 [1760/1949 (90%)]	Loss: 0.434484
Train Epoch: 4 [1920/1949 (98%)]	Loss: 2.044342

Test set: Average loss: 0.5249, Accuracy: 76/100 (76%)

Train Epoch: 5 [0/1949 (0%)]	Loss: 1.125168
Train Epoch: 5 [160/1949 (8%)]	Loss: 0.487420
Train Epoch: 5 [320/1949 (16%)]	Loss: 0.643658
Train Epoch: 5 [480/1949 (25%)]	Loss: 0.310708
Train Epoch: 5 [640/1949 (33%)]	Loss: 0.223017
Train Epoch: 5 [800/1949 (41%)]	Loss: 0.347709
Train Epoch: 5 [960/1949 (49%)]	Loss: 0.405673
Train Epoch: 5 [1120/1949 (57%)]	Loss: 0.509427
Train Epoch: 5 [1280/1949 (66%)]	Loss: 0.764386
Train Epoch: 5 [1440/1949 (74%)]	Loss: 0.342990
Train Epoch: 5 [1600/1949 (82%)]	Loss: 0.228474
Train Epoch: 5 [1760/1949 (90%)]	Loss: 0.671158
Train Epoch: 5 [1920/1949 (98%)]	Loss: 0.351810

Test set: Average loss: 0.5222, Accuracy: 83/100 (83%)

Train Epoch: 6 [0/1949 (0%)]	Loss: 0.246740
Train Epoch: 6 [160/1949 (8%)]	Loss: 0.119550
Train Epoch: 6 [320/1949 (16%)]	Loss: 1.007236
Train Epoch: 6 [480/1949 (25%)]	Loss: 1.279844
Train Epoch: 6 [640/1949 (33%)]	Loss: 0.407338
Train Epoch: 6 [800/1949 (41%)]	Loss: 0.098704
Train Epoch: 6 [960/1949 (49%)]	Loss: 0.963513
Train Epoch: 6 [1120/1949 (57%)]	Loss: 0.864720
Train Epoch: 6 [1280/1949 (66%)]	Loss: 0.617749
Train Epoch: 6 [1440/1949 (74%)]	Loss: 0.179300
Train Epoch: 6 [1600/1949 (82%)]	Loss: 0.509474
Train Epoch: 6 [1760/1949 (90%)]	Loss: 0.314553
Train Epoch: 6 [1920/1949 (98%)]	Loss: 0.402076

Test set: Average loss: 0.4075, Accuracy: 85/100 (85%)

Train Epoch: 7 [0/1949 (0%)]	Loss: 0.230927
Train Epoch: 7 [160/1949 (8%)]	Loss: 0.354949
Train Epoch: 7 [320/1949 (16%)]	Loss: 0.230590
Train Epoch: 7 [480/1949 (25%)]	Loss: 0.167888
Train Epoch: 7 [640/1949 (33%)]	Loss: 0.230623
Train Epoch: 7 [800/1949 (41%)]	Loss: 0.429072
Train Epoch: 7 [960/1949 (49%)]	Loss: 0.164219
Train Epoch: 7 [1120/1949 (57%)]	Loss: 0.409828
Train Epoch: 7 [1280/1949 (66%)]	Loss: 0.325223
Train Epoch: 7 [1440/1949 (74%)]	Loss: 0.252868
Train Epoch: 7 [1600/1949 (82%)]	Loss: 0.337788
Train Epoch: 7 [1760/1949 (90%)]	Loss: 0.221320
Train Epoch: 7 [1920/1949 (98%)]	Loss: 0.335692

Test set: Average loss: 0.4037, Accuracy: 83/100 (83%)

Train Epoch: 8 [0/1949 (0%)]	Loss: 0.080534
Train Epoch: 8 [160/1949 (8%)]	Loss: 0.704726
Train Epoch: 8 [320/1949 (16%)]	Loss: 0.097114
Train Epoch: 8 [480/1949 (25%)]	Loss: 0.091729
Train Epoch: 8 [640/1949 (33%)]	Loss: 0.056717
Train Epoch: 8 [800/1949 (41%)]	Loss: 0.088866
Train Epoch: 8 [960/1949 (49%)]	Loss: 0.133483
Train Epoch: 8 [1120/1949 (57%)]	Loss: 0.193675
Train Epoch: 8 [1280/1949 (66%)]	Loss: 0.091832
Train Epoch: 8 [1440/1949 (74%)]	Loss: 0.150204
Train Epoch: 8 [1600/1949 (82%)]	Loss: 0.146745
Train Epoch: 8 [1760/1949 (90%)]	Loss: 0.375873
Train Epoch: 8 [1920/1949 (98%)]	Loss: 0.299561

Test set: Average loss: 0.3427, Accuracy: 90/100 (90%)

Train Epoch: 9 [0/1949 (0%)]	Loss: 0.075616
Train Epoch: 9 [160/1949 (8%)]	Loss: 0.069953
Train Epoch: 9 [320/1949 (16%)]	Loss: 0.106683
Train Epoch: 9 [480/1949 (25%)]	Loss: 0.155001
Train Epoch: 9 [640/1949 (33%)]	Loss: 0.074617
Train Epoch: 9 [800/1949 (41%)]	Loss: 0.220201
Train Epoch: 9 [960/1949 (49%)]	Loss: 0.439087
Train Epoch: 9 [1120/1949 (57%)]	Loss: 0.324376
Train Epoch: 9 [1280/1949 (66%)]	Loss: 0.059005
Train Epoch: 9 [1440/1949 (74%)]	Loss: 0.180084
Train Epoch: 9 [1600/1949 (82%)]	Loss: 0.144776
Train Epoch: 9 [1760/1949 (90%)]	Loss: 0.148000
Train Epoch: 9 [1920/1949 (98%)]	Loss: 0.172169

Test set: Average loss: 0.3738, Accuracy: 89/100 (89%)
