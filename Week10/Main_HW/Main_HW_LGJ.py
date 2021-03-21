from __future__ import print_function
from torch import nn, optim, cuda
from torch.utils import data
from torchvision import datasets, transforms as transforms
import torch
import torchvision
import torch.nn.functional as F
import time

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# Training settings
batch_size = 64

device = 'cuda' if cuda.is_available() else 'cpu'
print(f'Training CIFAR10 Model on {device}\n{"=" * 44}')

# MNIST Dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True,
                                             transform=transforms.ToTensor())

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = data.DataLoader(dataset=test_dataset,
                              batch_size=batch_size,
                              shuffle=False)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(3072, 2000)
        self.l2 = nn.Linear(2000, 1400)
        self.l3 = nn.Linear(1400, 1000)
        self.l4 = nn.Linear(1000, 600)
        self.l5 = nn.Linear(600, 350)
        self.l6 = nn.Linear(350, 120)
        self.l7 = nn.Linear(120, 10)

    def forward(self, x):
        x = x.view(-1, 3072)  # Flatten the data (n, 1, 28, 28)-> (n, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        return self.l4(x)


model = Net()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.5)


def train(epoch):
    model.train()
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


def test():
    model.eval()
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
    print(f'===========================\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.0f}%)')


if __name__ == '__main__':
    since = time.time()
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
      
    
    "C:\Users\GIJIN LEE\anaconda3\envs\pytorch\python.exe" C:/source/PythonAI/Practice.py
Training CIFAR10 Model on cuda
============================================
Files already downloaded and verified
Train Epoch: 1 | Batch Status: 0/50000 (0%) | Loss: 6.398051
Train Epoch: 1 | Batch Status: 640/50000 (1%) | Loss: 3.992432
Train Epoch: 1 | Batch Status: 1280/50000 (3%) | Loss: 3.220172
Train Epoch: 1 | Batch Status: 1920/50000 (4%) | Loss: 2.410102
Train Epoch: 1 | Batch Status: 2560/50000 (5%) | Loss: 2.403196
Train Epoch: 1 | Batch Status: 3200/50000 (6%) | Loss: 2.278409
Train Epoch: 1 | Batch Status: 3840/50000 (8%) | Loss: 2.223634
Train Epoch: 1 | Batch Status: 4480/50000 (9%) | Loss: 2.231646
Train Epoch: 1 | Batch Status: 5120/50000 (10%) | Loss: 2.301297
Train Epoch: 1 | Batch Status: 5760/50000 (12%) | Loss: 2.321029
Train Epoch: 1 | Batch Status: 6400/50000 (13%) | Loss: 2.338765
Train Epoch: 1 | Batch Status: 7040/50000 (14%) | Loss: 2.226198
Train Epoch: 1 | Batch Status: 7680/50000 (15%) | Loss: 2.096586
Train Epoch: 1 | Batch Status: 8320/50000 (17%) | Loss: 2.397206
Train Epoch: 1 | Batch Status: 8960/50000 (18%) | Loss: 2.186212
Train Epoch: 1 | Batch Status: 9600/50000 (19%) | Loss: 2.033500
Train Epoch: 1 | Batch Status: 10240/50000 (20%) | Loss: 2.102834
Train Epoch: 1 | Batch Status: 10880/50000 (22%) | Loss: 2.173447
Train Epoch: 1 | Batch Status: 11520/50000 (23%) | Loss: 2.024846
Train Epoch: 1 | Batch Status: 12160/50000 (24%) | Loss: 2.156905
Train Epoch: 1 | Batch Status: 12800/50000 (26%) | Loss: 2.285615
Train Epoch: 1 | Batch Status: 13440/50000 (27%) | Loss: 2.061122
Train Epoch: 1 | Batch Status: 14080/50000 (28%) | Loss: 2.106953
Train Epoch: 1 | Batch Status: 14720/50000 (29%) | Loss: 2.120251
Train Epoch: 1 | Batch Status: 15360/50000 (31%) | Loss: 1.877983
Train Epoch: 1 | Batch Status: 16000/50000 (32%) | Loss: 1.885795
Train Epoch: 1 | Batch Status: 16640/50000 (33%) | Loss: 2.104372
Train Epoch: 1 | Batch Status: 17280/50000 (35%) | Loss: 1.868391
Train Epoch: 1 | Batch Status: 17920/50000 (36%) | Loss: 1.934703
Train Epoch: 1 | Batch Status: 18560/50000 (37%) | Loss: 2.110856
Train Epoch: 1 | Batch Status: 19200/50000 (38%) | Loss: 1.986352
Train Epoch: 1 | Batch Status: 19840/50000 (40%) | Loss: 2.039594
Train Epoch: 1 | Batch Status: 20480/50000 (41%) | Loss: 2.072969
Train Epoch: 1 | Batch Status: 21120/50000 (42%) | Loss: 1.979910
Train Epoch: 1 | Batch Status: 21760/50000 (43%) | Loss: 2.073531
Train Epoch: 1 | Batch Status: 22400/50000 (45%) | Loss: 1.853740
Train Epoch: 1 | Batch Status: 23040/50000 (46%) | Loss: 2.044943
Train Epoch: 1 | Batch Status: 23680/50000 (47%) | Loss: 1.565816
Train Epoch: 1 | Batch Status: 24320/50000 (49%) | Loss: 2.109794
Train Epoch: 1 | Batch Status: 24960/50000 (50%) | Loss: 1.782082
Train Epoch: 1 | Batch Status: 25600/50000 (51%) | Loss: 1.868243
Train Epoch: 1 | Batch Status: 26240/50000 (52%) | Loss: 1.945065
Train Epoch: 1 | Batch Status: 26880/50000 (54%) | Loss: 2.002161
Train Epoch: 1 | Batch Status: 27520/50000 (55%) | Loss: 1.822584
Train Epoch: 1 | Batch Status: 28160/50000 (56%) | Loss: 1.933394
Train Epoch: 1 | Batch Status: 28800/50000 (58%) | Loss: 2.126630
Train Epoch: 1 | Batch Status: 29440/50000 (59%) | Loss: 1.972499
Train Epoch: 1 | Batch Status: 30080/50000 (60%) | Loss: 1.935136
Train Epoch: 1 | Batch Status: 30720/50000 (61%) | Loss: 1.915297
Train Epoch: 1 | Batch Status: 31360/50000 (63%) | Loss: 1.903348
Train Epoch: 1 | Batch Status: 32000/50000 (64%) | Loss: 1.773728
Train Epoch: 1 | Batch Status: 32640/50000 (65%) | Loss: 1.734620
Train Epoch: 1 | Batch Status: 33280/50000 (66%) | Loss: 1.637045
Train Epoch: 1 | Batch Status: 33920/50000 (68%) | Loss: 2.098146
Train Epoch: 1 | Batch Status: 34560/50000 (69%) | Loss: 2.084217
Train Epoch: 1 | Batch Status: 35200/50000 (70%) | Loss: 2.026422
Train Epoch: 1 | Batch Status: 35840/50000 (72%) | Loss: 1.650345
Train Epoch: 1 | Batch Status: 36480/50000 (73%) | Loss: 1.933247
Train Epoch: 1 | Batch Status: 37120/50000 (74%) | Loss: 2.171828
Train Epoch: 1 | Batch Status: 37760/50000 (75%) | Loss: 1.833048
Train Epoch: 1 | Batch Status: 38400/50000 (77%) | Loss: 1.855304
Train Epoch: 1 | Batch Status: 39040/50000 (78%) | Loss: 1.942749
Train Epoch: 1 | Batch Status: 39680/50000 (79%) | Loss: 1.912119
Train Epoch: 1 | Batch Status: 40320/50000 (81%) | Loss: 1.915801
Train Epoch: 1 | Batch Status: 40960/50000 (82%) | Loss: 1.587759
Train Epoch: 1 | Batch Status: 41600/50000 (83%) | Loss: 1.981206
Train Epoch: 1 | Batch Status: 42240/50000 (84%) | Loss: 1.843709
Train Epoch: 1 | Batch Status: 42880/50000 (86%) | Loss: 1.712371
Train Epoch: 1 | Batch Status: 43520/50000 (87%) | Loss: 1.863550
Train Epoch: 1 | Batch Status: 44160/50000 (88%) | Loss: 2.002576
Train Epoch: 1 | Batch Status: 44800/50000 (90%) | Loss: 1.872329
Train Epoch: 1 | Batch Status: 45440/50000 (91%) | Loss: 1.930238
Train Epoch: 1 | Batch Status: 46080/50000 (92%) | Loss: 1.888922
Train Epoch: 1 | Batch Status: 46720/50000 (93%) | Loss: 1.690300
Train Epoch: 1 | Batch Status: 47360/50000 (95%) | Loss: 1.593166
Train Epoch: 1 | Batch Status: 48000/50000 (96%) | Loss: 1.941218
Train Epoch: 1 | Batch Status: 48640/50000 (97%) | Loss: 1.899793
Train Epoch: 1 | Batch Status: 49280/50000 (98%) | Loss: 1.850723
Train Epoch: 1 | Batch Status: 49920/50000 (100%) | Loss: 1.839771
Training time: 0m 10s
===========================
Test set: Average loss: 0.0315, Accuracy: 2906/10000 (29%)
Testing time: 0m 12s
Train Epoch: 2 | Batch Status: 0/50000 (0%) | Loss: 2.230531
Train Epoch: 2 | Batch Status: 640/50000 (1%) | Loss: 1.882361
Train Epoch: 2 | Batch Status: 1280/50000 (3%) | Loss: 1.629436
Train Epoch: 2 | Batch Status: 1920/50000 (4%) | Loss: 1.974381
Train Epoch: 2 | Batch Status: 2560/50000 (5%) | Loss: 1.822912
Train Epoch: 2 | Batch Status: 3200/50000 (6%) | Loss: 1.820745
Train Epoch: 2 | Batch Status: 3840/50000 (8%) | Loss: 1.631541
Train Epoch: 2 | Batch Status: 4480/50000 (9%) | Loss: 2.044153
Train Epoch: 2 | Batch Status: 5120/50000 (10%) | Loss: 1.715668
Train Epoch: 2 | Batch Status: 5760/50000 (12%) | Loss: 1.699191
Train Epoch: 2 | Batch Status: 6400/50000 (13%) | Loss: 1.823273
Train Epoch: 2 | Batch Status: 7040/50000 (14%) | Loss: 1.738961
Train Epoch: 2 | Batch Status: 7680/50000 (15%) | Loss: 1.801288
Train Epoch: 2 | Batch Status: 8320/50000 (17%) | Loss: 1.939911
Train Epoch: 2 | Batch Status: 8960/50000 (18%) | Loss: 1.661476
Train Epoch: 2 | Batch Status: 9600/50000 (19%) | Loss: 1.850001
Train Epoch: 2 | Batch Status: 10240/50000 (20%) | Loss: 1.680823
Train Epoch: 2 | Batch Status: 10880/50000 (22%) | Loss: 1.715240
Train Epoch: 2 | Batch Status: 11520/50000 (23%) | Loss: 2.123677
Train Epoch: 2 | Batch Status: 12160/50000 (24%) | Loss: 1.591145
Train Epoch: 2 | Batch Status: 12800/50000 (26%) | Loss: 1.788980
Train Epoch: 2 | Batch Status: 13440/50000 (27%) | Loss: 1.713180
Train Epoch: 2 | Batch Status: 14080/50000 (28%) | Loss: 1.730320
Train Epoch: 2 | Batch Status: 14720/50000 (29%) | Loss: 1.832708
Train Epoch: 2 | Batch Status: 15360/50000 (31%) | Loss: 1.682214
Train Epoch: 2 | Batch Status: 16000/50000 (32%) | Loss: 1.560855
Train Epoch: 2 | Batch Status: 16640/50000 (33%) | Loss: 1.720182
Train Epoch: 2 | Batch Status: 17280/50000 (35%) | Loss: 1.657699
Train Epoch: 2 | Batch Status: 17920/50000 (36%) | Loss: 1.839805
Train Epoch: 2 | Batch Status: 18560/50000 (37%) | Loss: 1.609064
Train Epoch: 2 | Batch Status: 19200/50000 (38%) | Loss: 1.850950
Train Epoch: 2 | Batch Status: 19840/50000 (40%) | Loss: 1.696032
Train Epoch: 2 | Batch Status: 20480/50000 (41%) | Loss: 1.924207
Train Epoch: 2 | Batch Status: 21120/50000 (42%) | Loss: 1.842718
Train Epoch: 2 | Batch Status: 21760/50000 (43%) | Loss: 1.805035
Train Epoch: 2 | Batch Status: 22400/50000 (45%) | Loss: 1.674559
Train Epoch: 2 | Batch Status: 23040/50000 (46%) | Loss: 1.678014
Train Epoch: 2 | Batch Status: 23680/50000 (47%) | Loss: 1.515996
Train Epoch: 2 | Batch Status: 24320/50000 (49%) | Loss: 1.754748
Train Epoch: 2 | Batch Status: 24960/50000 (50%) | Loss: 1.773107
Train Epoch: 2 | Batch Status: 25600/50000 (51%) | Loss: 1.693219
Train Epoch: 2 | Batch Status: 26240/50000 (52%) | Loss: 1.617722
Train Epoch: 2 | Batch Status: 26880/50000 (54%) | Loss: 1.993824
Train Epoch: 2 | Batch Status: 27520/50000 (55%) | Loss: 1.747096
Train Epoch: 2 | Batch Status: 28160/50000 (56%) | Loss: 1.673192
Train Epoch: 2 | Batch Status: 28800/50000 (58%) | Loss: 1.694546
Train Epoch: 2 | Batch Status: 29440/50000 (59%) | Loss: 1.606668
Train Epoch: 2 | Batch Status: 30080/50000 (60%) | Loss: 1.776567
Train Epoch: 2 | Batch Status: 30720/50000 (61%) | Loss: 1.721316
Train Epoch: 2 | Batch Status: 31360/50000 (63%) | Loss: 1.682265
Train Epoch: 2 | Batch Status: 32000/50000 (64%) | Loss: 1.590740
Train Epoch: 2 | Batch Status: 32640/50000 (65%) | Loss: 1.784169
Train Epoch: 2 | Batch Status: 33280/50000 (66%) | Loss: 1.889017
Train Epoch: 2 | Batch Status: 33920/50000 (68%) | Loss: 1.772183
Train Epoch: 2 | Batch Status: 34560/50000 (69%) | Loss: 1.545797
Train Epoch: 2 | Batch Status: 35200/50000 (70%) | Loss: 1.705222
Train Epoch: 2 | Batch Status: 35840/50000 (72%) | Loss: 1.824133
Train Epoch: 2 | Batch Status: 36480/50000 (73%) | Loss: 1.744232
Train Epoch: 2 | Batch Status: 37120/50000 (74%) | Loss: 1.563616
Train Epoch: 2 | Batch Status: 37760/50000 (75%) | Loss: 1.513323
Train Epoch: 2 | Batch Status: 38400/50000 (77%) | Loss: 1.664027
Train Epoch: 2 | Batch Status: 39040/50000 (78%) | Loss: 1.633644
Train Epoch: 2 | Batch Status: 39680/50000 (79%) | Loss: 1.505019
Train Epoch: 2 | Batch Status: 40320/50000 (81%) | Loss: 1.680231
Train Epoch: 2 | Batch Status: 40960/50000 (82%) | Loss: 1.854205
Train Epoch: 2 | Batch Status: 41600/50000 (83%) | Loss: 1.617872
Train Epoch: 2 | Batch Status: 42240/50000 (84%) | Loss: 1.623190
Train Epoch: 2 | Batch Status: 42880/50000 (86%) | Loss: 1.493168
Train Epoch: 2 | Batch Status: 43520/50000 (87%) | Loss: 1.771386
Train Epoch: 2 | Batch Status: 44160/50000 (88%) | Loss: 1.464987
Train Epoch: 2 | Batch Status: 44800/50000 (90%) | Loss: 1.728349
Train Epoch: 2 | Batch Status: 45440/50000 (91%) | Loss: 1.809181
Train Epoch: 2 | Batch Status: 46080/50000 (92%) | Loss: 1.694660
Train Epoch: 2 | Batch Status: 46720/50000 (93%) | Loss: 2.125156
Train Epoch: 2 | Batch Status: 47360/50000 (95%) | Loss: 1.646013
Train Epoch: 2 | Batch Status: 48000/50000 (96%) | Loss: 1.561677
Train Epoch: 2 | Batch Status: 48640/50000 (97%) | Loss: 1.567955
Train Epoch: 2 | Batch Status: 49280/50000 (98%) | Loss: 1.936590
Train Epoch: 2 | Batch Status: 49920/50000 (100%) | Loss: 2.154595
Training time: 0m 9s
===========================
Test set: Average loss: 0.0275, Accuracy: 3598/10000 (36%)
Testing time: 0m 11s
Train Epoch: 3 | Batch Status: 0/50000 (0%) | Loss: 1.868496
Train Epoch: 3 | Batch Status: 640/50000 (1%) | Loss: 1.490724
Train Epoch: 3 | Batch Status: 1280/50000 (3%) | Loss: 1.478847
Train Epoch: 3 | Batch Status: 1920/50000 (4%) | Loss: 1.747772
Train Epoch: 3 | Batch Status: 2560/50000 (5%) | Loss: 1.656864
Train Epoch: 3 | Batch Status: 3200/50000 (6%) | Loss: 1.691536
Train Epoch: 3 | Batch Status: 3840/50000 (8%) | Loss: 1.543115
Train Epoch: 3 | Batch Status: 4480/50000 (9%) | Loss: 1.558482
Train Epoch: 3 | Batch Status: 5120/50000 (10%) | Loss: 1.642153
Train Epoch: 3 | Batch Status: 5760/50000 (12%) | Loss: 1.713440
Train Epoch: 3 | Batch Status: 6400/50000 (13%) | Loss: 1.803125
Train Epoch: 3 | Batch Status: 7040/50000 (14%) | Loss: 1.242597
Train Epoch: 3 | Batch Status: 7680/50000 (15%) | Loss: 1.467713
Train Epoch: 3 | Batch Status: 8320/50000 (17%) | Loss: 1.555909
Train Epoch: 3 | Batch Status: 8960/50000 (18%) | Loss: 1.692625
Train Epoch: 3 | Batch Status: 9600/50000 (19%) | Loss: 1.718121
Train Epoch: 3 | Batch Status: 10240/50000 (20%) | Loss: 1.640037
Train Epoch: 3 | Batch Status: 10880/50000 (22%) | Loss: 1.646407
Train Epoch: 3 | Batch Status: 11520/50000 (23%) | Loss: 1.668251
Train Epoch: 3 | Batch Status: 12160/50000 (24%) | Loss: 1.405528
Train Epoch: 3 | Batch Status: 12800/50000 (26%) | Loss: 1.729787
Train Epoch: 3 | Batch Status: 13440/50000 (27%) | Loss: 1.559637
Train Epoch: 3 | Batch Status: 14080/50000 (28%) | Loss: 1.634600
Train Epoch: 3 | Batch Status: 14720/50000 (29%) | Loss: 1.787704
Train Epoch: 3 | Batch Status: 15360/50000 (31%) | Loss: 1.854246
Train Epoch: 3 | Batch Status: 16000/50000 (32%) | Loss: 1.844676
Train Epoch: 3 | Batch Status: 16640/50000 (33%) | Loss: 1.483557
Train Epoch: 3 | Batch Status: 17280/50000 (35%) | Loss: 1.531842
Train Epoch: 3 | Batch Status: 17920/50000 (36%) | Loss: 1.673799
Train Epoch: 3 | Batch Status: 18560/50000 (37%) | Loss: 1.346274
Train Epoch: 3 | Batch Status: 19200/50000 (38%) | Loss: 1.513685
Train Epoch: 3 | Batch Status: 19840/50000 (40%) | Loss: 1.675845
Train Epoch: 3 | Batch Status: 20480/50000 (41%) | Loss: 1.371257
Train Epoch: 3 | Batch Status: 21120/50000 (42%) | Loss: 1.977141
Train Epoch: 3 | Batch Status: 21760/50000 (43%) | Loss: 1.457214
Train Epoch: 3 | Batch Status: 22400/50000 (45%) | Loss: 1.602471
Train Epoch: 3 | Batch Status: 23040/50000 (46%) | Loss: 1.722850
Train Epoch: 3 | Batch Status: 23680/50000 (47%) | Loss: 1.459256
Train Epoch: 3 | Batch Status: 24320/50000 (49%) | Loss: 1.539629
Train Epoch: 3 | Batch Status: 24960/50000 (50%) | Loss: 1.476905
Train Epoch: 3 | Batch Status: 25600/50000 (51%) | Loss: 1.487205
Train Epoch: 3 | Batch Status: 26240/50000 (52%) | Loss: 1.701994
Train Epoch: 3 | Batch Status: 26880/50000 (54%) | Loss: 1.714379
Train Epoch: 3 | Batch Status: 27520/50000 (55%) | Loss: 1.557516
Train Epoch: 3 | Batch Status: 28160/50000 (56%) | Loss: 1.672955
Train Epoch: 3 | Batch Status: 28800/50000 (58%) | Loss: 1.568126
Train Epoch: 3 | Batch Status: 29440/50000 (59%) | Loss: 1.475902
Train Epoch: 3 | Batch Status: 30080/50000 (60%) | Loss: 1.433985
Train Epoch: 3 | Batch Status: 30720/50000 (61%) | Loss: 1.640463
Train Epoch: 3 | Batch Status: 31360/50000 (63%) | Loss: 1.548989
Train Epoch: 3 | Batch Status: 32000/50000 (64%) | Loss: 1.306856
Train Epoch: 3 | Batch Status: 32640/50000 (65%) | Loss: 1.651535
Train Epoch: 3 | Batch Status: 33280/50000 (66%) | Loss: 1.571747
Train Epoch: 3 | Batch Status: 33920/50000 (68%) | Loss: 1.808425
Train Epoch: 3 | Batch Status: 34560/50000 (69%) | Loss: 1.578785
Train Epoch: 3 | Batch Status: 35200/50000 (70%) | Loss: 1.649967
Train Epoch: 3 | Batch Status: 35840/50000 (72%) | Loss: 1.552085
Train Epoch: 3 | Batch Status: 36480/50000 (73%) | Loss: 1.579572
Train Epoch: 3 | Batch Status: 37120/50000 (74%) | Loss: 1.421536
Train Epoch: 3 | Batch Status: 37760/50000 (75%) | Loss: 1.485508
Train Epoch: 3 | Batch Status: 38400/50000 (77%) | Loss: 1.442703
Train Epoch: 3 | Batch Status: 39040/50000 (78%) | Loss: 1.359421
Train Epoch: 3 | Batch Status: 39680/50000 (79%) | Loss: 1.432431
Train Epoch: 3 | Batch Status: 40320/50000 (81%) | Loss: 1.615467
Train Epoch: 3 | Batch Status: 40960/50000 (82%) | Loss: 1.635517
Train Epoch: 3 | Batch Status: 41600/50000 (83%) | Loss: 1.884505
Train Epoch: 3 | Batch Status: 42240/50000 (84%) | Loss: 1.556883
Train Epoch: 3 | Batch Status: 42880/50000 (86%) | Loss: 1.573430
Train Epoch: 3 | Batch Status: 43520/50000 (87%) | Loss: 1.636234
Train Epoch: 3 | Batch Status: 44160/50000 (88%) | Loss: 1.629179
Train Epoch: 3 | Batch Status: 44800/50000 (90%) | Loss: 1.502118
Train Epoch: 3 | Batch Status: 45440/50000 (91%) | Loss: 1.701547
Train Epoch: 3 | Batch Status: 46080/50000 (92%) | Loss: 1.870219
Train Epoch: 3 | Batch Status: 46720/50000 (93%) | Loss: 1.482911
Train Epoch: 3 | Batch Status: 47360/50000 (95%) | Loss: 1.618767
Train Epoch: 3 | Batch Status: 48000/50000 (96%) | Loss: 1.485895
Train Epoch: 3 | Batch Status: 48640/50000 (97%) | Loss: 1.735925
Train Epoch: 3 | Batch Status: 49280/50000 (98%) | Loss: 1.920118
Train Epoch: 3 | Batch Status: 49920/50000 (100%) | Loss: 1.877838
Training time: 0m 9s
===========================
Test set: Average loss: 0.0301, Accuracy: 3505/10000 (35%)
Testing time: 0m 11s
Train Epoch: 4 | Batch Status: 0/50000 (0%) | Loss: 1.934070
Train Epoch: 4 | Batch Status: 640/50000 (1%) | Loss: 1.517544
Train Epoch: 4 | Batch Status: 1280/50000 (3%) | Loss: 1.332113
Train Epoch: 4 | Batch Status: 1920/50000 (4%) | Loss: 1.639199
Train Epoch: 4 | Batch Status: 2560/50000 (5%) | Loss: 1.314628
Train Epoch: 4 | Batch Status: 3200/50000 (6%) | Loss: 1.558103
Train Epoch: 4 | Batch Status: 3840/50000 (8%) | Loss: 1.452086
Train Epoch: 4 | Batch Status: 4480/50000 (9%) | Loss: 1.429538
Train Epoch: 4 | Batch Status: 5120/50000 (10%) | Loss: 1.447353
Train Epoch: 4 | Batch Status: 5760/50000 (12%) | Loss: 1.519064
Train Epoch: 4 | Batch Status: 6400/50000 (13%) | Loss: 1.624195
Train Epoch: 4 | Batch Status: 7040/50000 (14%) | Loss: 1.333376
Train Epoch: 4 | Batch Status: 7680/50000 (15%) | Loss: 1.618808
Train Epoch: 4 | Batch Status: 8320/50000 (17%) | Loss: 1.613421
Train Epoch: 4 | Batch Status: 8960/50000 (18%) | Loss: 1.398958
Train Epoch: 4 | Batch Status: 9600/50000 (19%) | Loss: 1.487312
Train Epoch: 4 | Batch Status: 10240/50000 (20%) | Loss: 1.538045
Train Epoch: 4 | Batch Status: 10880/50000 (22%) | Loss: 1.300388
Train Epoch: 4 | Batch Status: 11520/50000 (23%) | Loss: 1.631056
Train Epoch: 4 | Batch Status: 12160/50000 (24%) | Loss: 1.774459
Train Epoch: 4 | Batch Status: 12800/50000 (26%) | Loss: 1.337399
Train Epoch: 4 | Batch Status: 13440/50000 (27%) | Loss: 1.709227
Train Epoch: 4 | Batch Status: 14080/50000 (28%) | Loss: 1.571263
Train Epoch: 4 | Batch Status: 14720/50000 (29%) | Loss: 1.545870
Train Epoch: 4 | Batch Status: 15360/50000 (31%) | Loss: 1.821524
Train Epoch: 4 | Batch Status: 16000/50000 (32%) | Loss: 1.638245
Train Epoch: 4 | Batch Status: 16640/50000 (33%) | Loss: 1.561022
Train Epoch: 4 | Batch Status: 17280/50000 (35%) | Loss: 1.523026
Train Epoch: 4 | Batch Status: 17920/50000 (36%) | Loss: 1.441346
Train Epoch: 4 | Batch Status: 18560/50000 (37%) | Loss: 1.618553
Train Epoch: 4 | Batch Status: 19200/50000 (38%) | Loss: 1.651568
Train Epoch: 4 | Batch Status: 19840/50000 (40%) | Loss: 1.714567
Train Epoch: 4 | Batch Status: 20480/50000 (41%) | Loss: 1.576997
Train Epoch: 4 | Batch Status: 21120/50000 (42%) | Loss: 1.765915
Train Epoch: 4 | Batch Status: 21760/50000 (43%) | Loss: 1.388579
Train Epoch: 4 | Batch Status: 22400/50000 (45%) | Loss: 1.557051
Train Epoch: 4 | Batch Status: 23040/50000 (46%) | Loss: 1.481080
Train Epoch: 4 | Batch Status: 23680/50000 (47%) | Loss: 1.542596
Train Epoch: 4 | Batch Status: 24320/50000 (49%) | Loss: 1.543485
Train Epoch: 4 | Batch Status: 24960/50000 (50%) | Loss: 1.768656
Train Epoch: 4 | Batch Status: 25600/50000 (51%) | Loss: 1.777257
Train Epoch: 4 | Batch Status: 26240/50000 (52%) | Loss: 1.353735
Train Epoch: 4 | Batch Status: 26880/50000 (54%) | Loss: 1.693515
Train Epoch: 4 | Batch Status: 27520/50000 (55%) | Loss: 1.689089
Train Epoch: 4 | Batch Status: 28160/50000 (56%) | Loss: 1.503943
Train Epoch: 4 | Batch Status: 28800/50000 (58%) | Loss: 1.397257
Train Epoch: 4 | Batch Status: 29440/50000 (59%) | Loss: 1.622011
Train Epoch: 4 | Batch Status: 30080/50000 (60%) | Loss: 1.429422
Train Epoch: 4 | Batch Status: 30720/50000 (61%) | Loss: 1.479007
Train Epoch: 4 | Batch Status: 31360/50000 (63%) | Loss: 1.641650
Train Epoch: 4 | Batch Status: 32000/50000 (64%) | Loss: 1.573541
Train Epoch: 4 | Batch Status: 32640/50000 (65%) | Loss: 1.426895
Train Epoch: 4 | Batch Status: 33280/50000 (66%) | Loss: 1.469681
Train Epoch: 4 | Batch Status: 33920/50000 (68%) | Loss: 1.474273
Train Epoch: 4 | Batch Status: 34560/50000 (69%) | Loss: 1.432456
Train Epoch: 4 | Batch Status: 35200/50000 (70%) | Loss: 1.537692
Train Epoch: 4 | Batch Status: 35840/50000 (72%) | Loss: 1.568468
Train Epoch: 4 | Batch Status: 36480/50000 (73%) | Loss: 1.776023
Train Epoch: 4 | Batch Status: 37120/50000 (74%) | Loss: 1.442248
Train Epoch: 4 | Batch Status: 37760/50000 (75%) | Loss: 1.749458
Train Epoch: 4 | Batch Status: 38400/50000 (77%) | Loss: 1.673341
Train Epoch: 4 | Batch Status: 39040/50000 (78%) | Loss: 1.652852
Train Epoch: 4 | Batch Status: 39680/50000 (79%) | Loss: 1.646102
Train Epoch: 4 | Batch Status: 40320/50000 (81%) | Loss: 1.454009
Train Epoch: 4 | Batch Status: 40960/50000 (82%) | Loss: 1.398226
Train Epoch: 4 | Batch Status: 41600/50000 (83%) | Loss: 1.398782
Train Epoch: 4 | Batch Status: 42240/50000 (84%) | Loss: 1.333098
Train Epoch: 4 | Batch Status: 42880/50000 (86%) | Loss: 1.468869
Train Epoch: 4 | Batch Status: 43520/50000 (87%) | Loss: 1.441612
Train Epoch: 4 | Batch Status: 44160/50000 (88%) | Loss: 1.293263
Train Epoch: 4 | Batch Status: 44800/50000 (90%) | Loss: 1.545864
Train Epoch: 4 | Batch Status: 45440/50000 (91%) | Loss: 1.475123
Train Epoch: 4 | Batch Status: 46080/50000 (92%) | Loss: 1.589944
Train Epoch: 4 | Batch Status: 46720/50000 (93%) | Loss: 1.541991
Train Epoch: 4 | Batch Status: 47360/50000 (95%) | Loss: 1.367499
Train Epoch: 4 | Batch Status: 48000/50000 (96%) | Loss: 1.581974
Train Epoch: 4 | Batch Status: 48640/50000 (97%) | Loss: 1.602936
Train Epoch: 4 | Batch Status: 49280/50000 (98%) | Loss: 1.435656
Train Epoch: 4 | Batch Status: 49920/50000 (100%) | Loss: 1.489182
Training time: 0m 9s
===========================
Test set: Average loss: 0.0238, Accuracy: 4590/10000 (46%)
Testing time: 0m 11s
Train Epoch: 5 | Batch Status: 0/50000 (0%) | Loss: 1.698536
Train Epoch: 5 | Batch Status: 640/50000 (1%) | Loss: 1.468698
Train Epoch: 5 | Batch Status: 1280/50000 (3%) | Loss: 1.670976
Train Epoch: 5 | Batch Status: 1920/50000 (4%) | Loss: 1.478557
Train Epoch: 5 | Batch Status: 2560/50000 (5%) | Loss: 1.558866
Train Epoch: 5 | Batch Status: 3200/50000 (6%) | Loss: 1.446986
Train Epoch: 5 | Batch Status: 3840/50000 (8%) | Loss: 1.451758
Train Epoch: 5 | Batch Status: 4480/50000 (9%) | Loss: 1.756257
Train Epoch: 5 | Batch Status: 5120/50000 (10%) | Loss: 1.423650
Train Epoch: 5 | Batch Status: 5760/50000 (12%) | Loss: 1.462201
Train Epoch: 5 | Batch Status: 6400/50000 (13%) | Loss: 1.409656
Train Epoch: 5 | Batch Status: 7040/50000 (14%) | Loss: 1.481037
Train Epoch: 5 | Batch Status: 7680/50000 (15%) | Loss: 1.415034
Train Epoch: 5 | Batch Status: 8320/50000 (17%) | Loss: 1.622687
Train Epoch: 5 | Batch Status: 8960/50000 (18%) | Loss: 1.377612
Train Epoch: 5 | Batch Status: 9600/50000 (19%) | Loss: 1.158798
Train Epoch: 5 | Batch Status: 10240/50000 (20%) | Loss: 1.503832
Train Epoch: 5 | Batch Status: 10880/50000 (22%) | Loss: 1.274684
Train Epoch: 5 | Batch Status: 11520/50000 (23%) | Loss: 1.469704
Train Epoch: 5 | Batch Status: 12160/50000 (24%) | Loss: 1.429538
Train Epoch: 5 | Batch Status: 12800/50000 (26%) | Loss: 1.398754
Train Epoch: 5 | Batch Status: 13440/50000 (27%) | Loss: 1.402130
Train Epoch: 5 | Batch Status: 14080/50000 (28%) | Loss: 1.362835
Train Epoch: 5 | Batch Status: 14720/50000 (29%) | Loss: 1.174755
Train Epoch: 5 | Batch Status: 15360/50000 (31%) | Loss: 1.494431
Train Epoch: 5 | Batch Status: 16000/50000 (32%) | Loss: 1.425043
Train Epoch: 5 | Batch Status: 16640/50000 (33%) | Loss: 1.569813
Train Epoch: 5 | Batch Status: 17280/50000 (35%) | Loss: 1.635064
Train Epoch: 5 | Batch Status: 17920/50000 (36%) | Loss: 1.678997
Train Epoch: 5 | Batch Status: 18560/50000 (37%) | Loss: 1.498713
Train Epoch: 5 | Batch Status: 19200/50000 (38%) | Loss: 1.347771
Train Epoch: 5 | Batch Status: 19840/50000 (40%) | Loss: 1.523927
Train Epoch: 5 | Batch Status: 20480/50000 (41%) | Loss: 1.300372
Train Epoch: 5 | Batch Status: 21120/50000 (42%) | Loss: 1.503080
Train Epoch: 5 | Batch Status: 21760/50000 (43%) | Loss: 1.448215
Train Epoch: 5 | Batch Status: 22400/50000 (45%) | Loss: 1.231366
Train Epoch: 5 | Batch Status: 23040/50000 (46%) | Loss: 1.452621
Train Epoch: 5 | Batch Status: 23680/50000 (47%) | Loss: 1.414839
Train Epoch: 5 | Batch Status: 24320/50000 (49%) | Loss: 1.348172
Train Epoch: 5 | Batch Status: 24960/50000 (50%) | Loss: 1.440692
Train Epoch: 5 | Batch Status: 25600/50000 (51%) | Loss: 1.530868
Train Epoch: 5 | Batch Status: 26240/50000 (52%) | Loss: 1.664453
Train Epoch: 5 | Batch Status: 26880/50000 (54%) | Loss: 1.768162
Train Epoch: 5 | Batch Status: 27520/50000 (55%) | Loss: 1.349911
Train Epoch: 5 | Batch Status: 28160/50000 (56%) | Loss: 1.433121
Train Epoch: 5 | Batch Status: 28800/50000 (58%) | Loss: 1.552829
Train Epoch: 5 | Batch Status: 29440/50000 (59%) | Loss: 1.374566
Train Epoch: 5 | Batch Status: 30080/50000 (60%) | Loss: 1.228611
Train Epoch: 5 | Batch Status: 30720/50000 (61%) | Loss: 1.479250
Train Epoch: 5 | Batch Status: 31360/50000 (63%) | Loss: 1.638279
Train Epoch: 5 | Batch Status: 32000/50000 (64%) | Loss: 1.461506
Train Epoch: 5 | Batch Status: 32640/50000 (65%) | Loss: 1.379125
Train Epoch: 5 | Batch Status: 33280/50000 (66%) | Loss: 1.447488
Train Epoch: 5 | Batch Status: 33920/50000 (68%) | Loss: 1.750650
Train Epoch: 5 | Batch Status: 34560/50000 (69%) | Loss: 1.137319
Train Epoch: 5 | Batch Status: 35200/50000 (70%) | Loss: 1.391259
Train Epoch: 5 | Batch Status: 35840/50000 (72%) | Loss: 1.384944
Train Epoch: 5 | Batch Status: 36480/50000 (73%) | Loss: 1.600767
Train Epoch: 5 | Batch Status: 37120/50000 (74%) | Loss: 1.377571
Train Epoch: 5 | Batch Status: 37760/50000 (75%) | Loss: 1.762839
Train Epoch: 5 | Batch Status: 38400/50000 (77%) | Loss: 1.407819
Train Epoch: 5 | Batch Status: 39040/50000 (78%) | Loss: 1.356114
Train Epoch: 5 | Batch Status: 39680/50000 (79%) | Loss: 1.469888
Train Epoch: 5 | Batch Status: 40320/50000 (81%) | Loss: 1.289846
Train Epoch: 5 | Batch Status: 40960/50000 (82%) | Loss: 1.393967
Train Epoch: 5 | Batch Status: 41600/50000 (83%) | Loss: 1.410797
Train Epoch: 5 | Batch Status: 42240/50000 (84%) | Loss: 1.455772
Train Epoch: 5 | Batch Status: 42880/50000 (86%) | Loss: 1.595643
Train Epoch: 5 | Batch Status: 43520/50000 (87%) | Loss: 1.422187
Train Epoch: 5 | Batch Status: 44160/50000 (88%) | Loss: 1.431812
Train Epoch: 5 | Batch Status: 44800/50000 (90%) | Loss: 1.386300
Train Epoch: 5 | Batch Status: 45440/50000 (91%) | Loss: 1.834135
Train Epoch: 5 | Batch Status: 46080/50000 (92%) | Loss: 1.583595
Train Epoch: 5 | Batch Status: 46720/50000 (93%) | Loss: 1.278124
Train Epoch: 5 | Batch Status: 47360/50000 (95%) | Loss: 1.472345
Train Epoch: 5 | Batch Status: 48000/50000 (96%) | Loss: 1.432263
Train Epoch: 5 | Batch Status: 48640/50000 (97%) | Loss: 1.532499
Train Epoch: 5 | Batch Status: 49280/50000 (98%) | Loss: 1.284242
Train Epoch: 5 | Batch Status: 49920/50000 (100%) | Loss: 1.615844
Training time: 0m 9s
===========================
Test set: Average loss: 0.0245, Accuracy: 4429/10000 (44%)
Testing time: 0m 11s
Train Epoch: 6 | Batch Status: 0/50000 (0%) | Loss: 1.339960
Train Epoch: 6 | Batch Status: 640/50000 (1%) | Loss: 1.571990
Train Epoch: 6 | Batch Status: 1280/50000 (3%) | Loss: 1.301767
Train Epoch: 6 | Batch Status: 1920/50000 (4%) | Loss: 1.621605
Train Epoch: 6 | Batch Status: 2560/50000 (5%) | Loss: 1.388628
Train Epoch: 6 | Batch Status: 3200/50000 (6%) | Loss: 1.156270
Train Epoch: 6 | Batch Status: 3840/50000 (8%) | Loss: 1.594032
Train Epoch: 6 | Batch Status: 4480/50000 (9%) | Loss: 1.470057
Train Epoch: 6 | Batch Status: 5120/50000 (10%) | Loss: 1.418723
Train Epoch: 6 | Batch Status: 5760/50000 (12%) | Loss: 1.487832
Train Epoch: 6 | Batch Status: 6400/50000 (13%) | Loss: 1.832397
Train Epoch: 6 | Batch Status: 7040/50000 (14%) | Loss: 1.403528
Train Epoch: 6 | Batch Status: 7680/50000 (15%) | Loss: 1.267401
Train Epoch: 6 | Batch Status: 8320/50000 (17%) | Loss: 1.519943
Train Epoch: 6 | Batch Status: 8960/50000 (18%) | Loss: 1.443209
Train Epoch: 6 | Batch Status: 9600/50000 (19%) | Loss: 1.594941
Train Epoch: 6 | Batch Status: 10240/50000 (20%) | Loss: 1.343695
Train Epoch: 6 | Batch Status: 10880/50000 (22%) | Loss: 1.401546
Train Epoch: 6 | Batch Status: 11520/50000 (23%) | Loss: 1.327198
Train Epoch: 6 | Batch Status: 12160/50000 (24%) | Loss: 1.303486
Train Epoch: 6 | Batch Status: 12800/50000 (26%) | Loss: 1.447997
Train Epoch: 6 | Batch Status: 13440/50000 (27%) | Loss: 1.233955
Train Epoch: 6 | Batch Status: 14080/50000 (28%) | Loss: 1.490499
Train Epoch: 6 | Batch Status: 14720/50000 (29%) | Loss: 1.588006
Train Epoch: 6 | Batch Status: 15360/50000 (31%) | Loss: 1.334167
Train Epoch: 6 | Batch Status: 16000/50000 (32%) | Loss: 1.477793
Train Epoch: 6 | Batch Status: 16640/50000 (33%) | Loss: 1.171768
Train Epoch: 6 | Batch Status: 17280/50000 (35%) | Loss: 1.355320
Train Epoch: 6 | Batch Status: 17920/50000 (36%) | Loss: 1.407955
Train Epoch: 6 | Batch Status: 18560/50000 (37%) | Loss: 1.186089
Train Epoch: 6 | Batch Status: 19200/50000 (38%) | Loss: 1.358789
Train Epoch: 6 | Batch Status: 19840/50000 (40%) | Loss: 1.281003
Train Epoch: 6 | Batch Status: 20480/50000 (41%) | Loss: 1.402234
Train Epoch: 6 | Batch Status: 21120/50000 (42%) | Loss: 1.284671
Train Epoch: 6 | Batch Status: 21760/50000 (43%) | Loss: 1.406825
Train Epoch: 6 | Batch Status: 22400/50000 (45%) | Loss: 1.492279
Train Epoch: 6 | Batch Status: 23040/50000 (46%) | Loss: 1.321072
Train Epoch: 6 | Batch Status: 23680/50000 (47%) | Loss: 1.298183
Train Epoch: 6 | Batch Status: 24320/50000 (49%) | Loss: 1.487882
Train Epoch: 6 | Batch Status: 24960/50000 (50%) | Loss: 1.541296
Train Epoch: 6 | Batch Status: 25600/50000 (51%) | Loss: 1.477683
Train Epoch: 6 | Batch Status: 26240/50000 (52%) | Loss: 1.353392
Train Epoch: 6 | Batch Status: 26880/50000 (54%) | Loss: 1.393588
Train Epoch: 6 | Batch Status: 27520/50000 (55%) | Loss: 1.286203
Train Epoch: 6 | Batch Status: 28160/50000 (56%) | Loss: 1.466236
Train Epoch: 6 | Batch Status: 28800/50000 (58%) | Loss: 1.440215
Train Epoch: 6 | Batch Status: 29440/50000 (59%) | Loss: 1.354211
Train Epoch: 6 | Batch Status: 30080/50000 (60%) | Loss: 1.444768
Train Epoch: 6 | Batch Status: 30720/50000 (61%) | Loss: 1.294494
Train Epoch: 6 | Batch Status: 31360/50000 (63%) | Loss: 1.343122
Train Epoch: 6 | Batch Status: 32000/50000 (64%) | Loss: 1.688097
Train Epoch: 6 | Batch Status: 32640/50000 (65%) | Loss: 1.667102
Train Epoch: 6 | Batch Status: 33280/50000 (66%) | Loss: 1.591313
Train Epoch: 6 | Batch Status: 33920/50000 (68%) | Loss: 1.342554
Train Epoch: 6 | Batch Status: 34560/50000 (69%) | Loss: 1.266715
Train Epoch: 6 | Batch Status: 35200/50000 (70%) | Loss: 1.436674
Train Epoch: 6 | Batch Status: 35840/50000 (72%) | Loss: 1.312528
Train Epoch: 6 | Batch Status: 36480/50000 (73%) | Loss: 1.291919
Train Epoch: 6 | Batch Status: 37120/50000 (74%) | Loss: 1.427259
Train Epoch: 6 | Batch Status: 37760/50000 (75%) | Loss: 1.071354
Train Epoch: 6 | Batch Status: 38400/50000 (77%) | Loss: 1.369319
Train Epoch: 6 | Batch Status: 39040/50000 (78%) | Loss: 1.395184
Train Epoch: 6 | Batch Status: 39680/50000 (79%) | Loss: 1.370884
Train Epoch: 6 | Batch Status: 40320/50000 (81%) | Loss: 1.449053
Train Epoch: 6 | Batch Status: 40960/50000 (82%) | Loss: 1.208967
Train Epoch: 6 | Batch Status: 41600/50000 (83%) | Loss: 1.383254
Train Epoch: 6 | Batch Status: 42240/50000 (84%) | Loss: 1.501489
Train Epoch: 6 | Batch Status: 42880/50000 (86%) | Loss: 1.545686
Train Epoch: 6 | Batch Status: 43520/50000 (87%) | Loss: 1.162090
Train Epoch: 6 | Batch Status: 44160/50000 (88%) | Loss: 1.541584
Train Epoch: 6 | Batch Status: 44800/50000 (90%) | Loss: 1.319664
Train Epoch: 6 | Batch Status: 45440/50000 (91%) | Loss: 1.543970
Train Epoch: 6 | Batch Status: 46080/50000 (92%) | Loss: 1.401636
Train Epoch: 6 | Batch Status: 46720/50000 (93%) | Loss: 1.270535
Train Epoch: 6 | Batch Status: 47360/50000 (95%) | Loss: 1.589585
Train Epoch: 6 | Batch Status: 48000/50000 (96%) | Loss: 1.389435
Train Epoch: 6 | Batch Status: 48640/50000 (97%) | Loss: 1.274673
Train Epoch: 6 | Batch Status: 49280/50000 (98%) | Loss: 1.893343
Train Epoch: 6 | Batch Status: 49920/50000 (100%) | Loss: 1.250669
Training time: 0m 9s
===========================
Test set: Average loss: 0.0236, Accuracy: 4574/10000 (46%)
Testing time: 0m 11s
Train Epoch: 7 | Batch Status: 0/50000 (0%) | Loss: 1.662635
Train Epoch: 7 | Batch Status: 640/50000 (1%) | Loss: 1.260646
Train Epoch: 7 | Batch Status: 1280/50000 (3%) | Loss: 1.414179
Train Epoch: 7 | Batch Status: 1920/50000 (4%) | Loss: 1.367745
Train Epoch: 7 | Batch Status: 2560/50000 (5%) | Loss: 1.444144
Train Epoch: 7 | Batch Status: 3200/50000 (6%) | Loss: 1.398016
Train Epoch: 7 | Batch Status: 3840/50000 (8%) | Loss: 1.362922
Train Epoch: 7 | Batch Status: 4480/50000 (9%) | Loss: 1.372878
Train Epoch: 7 | Batch Status: 5120/50000 (10%) | Loss: 1.300608
Train Epoch: 7 | Batch Status: 5760/50000 (12%) | Loss: 1.503664
Train Epoch: 7 | Batch Status: 6400/50000 (13%) | Loss: 1.591075
Train Epoch: 7 | Batch Status: 7040/50000 (14%) | Loss: 1.131292
Train Epoch: 7 | Batch Status: 7680/50000 (15%) | Loss: 1.332806
Train Epoch: 7 | Batch Status: 8320/50000 (17%) | Loss: 1.282299
Train Epoch: 7 | Batch Status: 8960/50000 (18%) | Loss: 1.325045
Train Epoch: 7 | Batch Status: 9600/50000 (19%) | Loss: 1.579155
Train Epoch: 7 | Batch Status: 10240/50000 (20%) | Loss: 1.490489
Train Epoch: 7 | Batch Status: 10880/50000 (22%) | Loss: 1.437288
Train Epoch: 7 | Batch Status: 11520/50000 (23%) | Loss: 1.373405
Train Epoch: 7 | Batch Status: 12160/50000 (24%) | Loss: 1.368219
Train Epoch: 7 | Batch Status: 12800/50000 (26%) | Loss: 1.370183
Train Epoch: 7 | Batch Status: 13440/50000 (27%) | Loss: 1.424564
Train Epoch: 7 | Batch Status: 14080/50000 (28%) | Loss: 1.328713
Train Epoch: 7 | Batch Status: 14720/50000 (29%) | Loss: 1.129758
Train Epoch: 7 | Batch Status: 15360/50000 (31%) | Loss: 1.382174
Train Epoch: 7 | Batch Status: 16000/50000 (32%) | Loss: 1.358608
Train Epoch: 7 | Batch Status: 16640/50000 (33%) | Loss: 1.400820
Train Epoch: 7 | Batch Status: 17280/50000 (35%) | Loss: 1.487541
Train Epoch: 7 | Batch Status: 17920/50000 (36%) | Loss: 1.308013
Train Epoch: 7 | Batch Status: 18560/50000 (37%) | Loss: 1.176255
Train Epoch: 7 | Batch Status: 19200/50000 (38%) | Loss: 1.268821
Train Epoch: 7 | Batch Status: 19840/50000 (40%) | Loss: 1.416732
Train Epoch: 7 | Batch Status: 20480/50000 (41%) | Loss: 1.351578
Train Epoch: 7 | Batch Status: 21120/50000 (42%) | Loss: 1.360443
Train Epoch: 7 | Batch Status: 21760/50000 (43%) | Loss: 1.510461
Train Epoch: 7 | Batch Status: 22400/50000 (45%) | Loss: 1.265527
Train Epoch: 7 | Batch Status: 23040/50000 (46%) | Loss: 1.413128
Train Epoch: 7 | Batch Status: 23680/50000 (47%) | Loss: 1.344486
Train Epoch: 7 | Batch Status: 24320/50000 (49%) | Loss: 1.259250
Train Epoch: 7 | Batch Status: 24960/50000 (50%) | Loss: 1.344874
Train Epoch: 7 | Batch Status: 25600/50000 (51%) | Loss: 1.741898
Train Epoch: 7 | Batch Status: 26240/50000 (52%) | Loss: 1.475353
Train Epoch: 7 | Batch Status: 26880/50000 (54%) | Loss: 1.269418
Train Epoch: 7 | Batch Status: 27520/50000 (55%) | Loss: 1.386551
Train Epoch: 7 | Batch Status: 28160/50000 (56%) | Loss: 1.355090
Train Epoch: 7 | Batch Status: 28800/50000 (58%) | Loss: 1.457179
Train Epoch: 7 | Batch Status: 29440/50000 (59%) | Loss: 1.265945
Train Epoch: 7 | Batch Status: 30080/50000 (60%) | Loss: 1.750060
Train Epoch: 7 | Batch Status: 30720/50000 (61%) | Loss: 1.505940
Train Epoch: 7 | Batch Status: 31360/50000 (63%) | Loss: 1.426243
Train Epoch: 7 | Batch Status: 32000/50000 (64%) | Loss: 1.257239
Train Epoch: 7 | Batch Status: 32640/50000 (65%) | Loss: 1.488055
Train Epoch: 7 | Batch Status: 33280/50000 (66%) | Loss: 1.504225
Train Epoch: 7 | Batch Status: 33920/50000 (68%) | Loss: 1.473150
Train Epoch: 7 | Batch Status: 34560/50000 (69%) | Loss: 1.314223
Train Epoch: 7 | Batch Status: 35200/50000 (70%) | Loss: 1.273371
Train Epoch: 7 | Batch Status: 35840/50000 (72%) | Loss: 1.069299
Train Epoch: 7 | Batch Status: 36480/50000 (73%) | Loss: 1.314752
Train Epoch: 7 | Batch Status: 37120/50000 (74%) | Loss: 1.454474
Train Epoch: 7 | Batch Status: 37760/50000 (75%) | Loss: 1.113086
Train Epoch: 7 | Batch Status: 38400/50000 (77%) | Loss: 1.306480
Train Epoch: 7 | Batch Status: 39040/50000 (78%) | Loss: 1.285692
Train Epoch: 7 | Batch Status: 39680/50000 (79%) | Loss: 1.313295
Train Epoch: 7 | Batch Status: 40320/50000 (81%) | Loss: 1.639576
Train Epoch: 7 | Batch Status: 40960/50000 (82%) | Loss: 1.215717
Train Epoch: 7 | Batch Status: 41600/50000 (83%) | Loss: 1.215430
Train Epoch: 7 | Batch Status: 42240/50000 (84%) | Loss: 1.463560
Train Epoch: 7 | Batch Status: 42880/50000 (86%) | Loss: 1.333831
Train Epoch: 7 | Batch Status: 43520/50000 (87%) | Loss: 1.374302
Train Epoch: 7 | Batch Status: 44160/50000 (88%) | Loss: 1.445661
Train Epoch: 7 | Batch Status: 44800/50000 (90%) | Loss: 1.400875
Train Epoch: 7 | Batch Status: 45440/50000 (91%) | Loss: 1.152466
Train Epoch: 7 | Batch Status: 46080/50000 (92%) | Loss: 1.489400
Train Epoch: 7 | Batch Status: 46720/50000 (93%) | Loss: 1.346172
Train Epoch: 7 | Batch Status: 47360/50000 (95%) | Loss: 1.323494
Train Epoch: 7 | Batch Status: 48000/50000 (96%) | Loss: 1.309828
Train Epoch: 7 | Batch Status: 48640/50000 (97%) | Loss: 1.186336
Train Epoch: 7 | Batch Status: 49280/50000 (98%) | Loss: 1.535088
Train Epoch: 7 | Batch Status: 49920/50000 (100%) | Loss: 1.150822
Training time: 0m 9s
===========================
Test set: Average loss: 0.0236, Accuracy: 4676/10000 (47%)
Testing time: 0m 11s
Train Epoch: 8 | Batch Status: 0/50000 (0%) | Loss: 1.332185
Train Epoch: 8 | Batch Status: 640/50000 (1%) | Loss: 1.403852
Train Epoch: 8 | Batch Status: 1280/50000 (3%) | Loss: 1.429609
Train Epoch: 8 | Batch Status: 1920/50000 (4%) | Loss: 1.500347
Train Epoch: 8 | Batch Status: 2560/50000 (5%) | Loss: 1.354776
Train Epoch: 8 | Batch Status: 3200/50000 (6%) | Loss: 1.362184
Train Epoch: 8 | Batch Status: 3840/50000 (8%) | Loss: 1.306084
Train Epoch: 8 | Batch Status: 4480/50000 (9%) | Loss: 1.312721
Train Epoch: 8 | Batch Status: 5120/50000 (10%) | Loss: 1.044087
Train Epoch: 8 | Batch Status: 5760/50000 (12%) | Loss: 1.101823
Train Epoch: 8 | Batch Status: 6400/50000 (13%) | Loss: 1.200871
Train Epoch: 8 | Batch Status: 7040/50000 (14%) | Loss: 1.329438
Train Epoch: 8 | Batch Status: 7680/50000 (15%) | Loss: 1.334058
Train Epoch: 8 | Batch Status: 8320/50000 (17%) | Loss: 1.236474
Train Epoch: 8 | Batch Status: 8960/50000 (18%) | Loss: 1.412931
Train Epoch: 8 | Batch Status: 9600/50000 (19%) | Loss: 1.216669
Train Epoch: 8 | Batch Status: 10240/50000 (20%) | Loss: 1.393962
Train Epoch: 8 | Batch Status: 10880/50000 (22%) | Loss: 1.276743
Train Epoch: 8 | Batch Status: 11520/50000 (23%) | Loss: 1.248166
Train Epoch: 8 | Batch Status: 12160/50000 (24%) | Loss: 1.474095
Train Epoch: 8 | Batch Status: 12800/50000 (26%) | Loss: 1.329180
Train Epoch: 8 | Batch Status: 13440/50000 (27%) | Loss: 1.378031
Train Epoch: 8 | Batch Status: 14080/50000 (28%) | Loss: 1.375841
Train Epoch: 8 | Batch Status: 14720/50000 (29%) | Loss: 1.288851
Train Epoch: 8 | Batch Status: 15360/50000 (31%) | Loss: 1.116530
Train Epoch: 8 | Batch Status: 16000/50000 (32%) | Loss: 1.389947
Train Epoch: 8 | Batch Status: 16640/50000 (33%) | Loss: 1.017659
Train Epoch: 8 | Batch Status: 17280/50000 (35%) | Loss: 1.319410
Train Epoch: 8 | Batch Status: 17920/50000 (36%) | Loss: 1.396248
Train Epoch: 8 | Batch Status: 18560/50000 (37%) | Loss: 1.187692
Train Epoch: 8 | Batch Status: 19200/50000 (38%) | Loss: 1.376235
Train Epoch: 8 | Batch Status: 19840/50000 (40%) | Loss: 1.185461
Train Epoch: 8 | Batch Status: 20480/50000 (41%) | Loss: 1.382204
Train Epoch: 8 | Batch Status: 21120/50000 (42%) | Loss: 1.154222
Train Epoch: 8 | Batch Status: 21760/50000 (43%) | Loss: 1.344824
Train Epoch: 8 | Batch Status: 22400/50000 (45%) | Loss: 1.279543
Train Epoch: 8 | Batch Status: 23040/50000 (46%) | Loss: 1.212722
Train Epoch: 8 | Batch Status: 23680/50000 (47%) | Loss: 1.272170
Train Epoch: 8 | Batch Status: 24320/50000 (49%) | Loss: 1.507506
Train Epoch: 8 | Batch Status: 24960/50000 (50%) | Loss: 1.329712
Train Epoch: 8 | Batch Status: 25600/50000 (51%) | Loss: 1.475921
Train Epoch: 8 | Batch Status: 26240/50000 (52%) | Loss: 1.437052
Train Epoch: 8 | Batch Status: 26880/50000 (54%) | Loss: 1.236145
Train Epoch: 8 | Batch Status: 27520/50000 (55%) | Loss: 1.160113
Train Epoch: 8 | Batch Status: 28160/50000 (56%) | Loss: 1.199183
Train Epoch: 8 | Batch Status: 28800/50000 (58%) | Loss: 1.529135
Train Epoch: 8 | Batch Status: 29440/50000 (59%) | Loss: 1.292437
Train Epoch: 8 | Batch Status: 30080/50000 (60%) | Loss: 1.476455
Train Epoch: 8 | Batch Status: 30720/50000 (61%) | Loss: 1.497761
Train Epoch: 8 | Batch Status: 31360/50000 (63%) | Loss: 1.209237
Train Epoch: 8 | Batch Status: 32000/50000 (64%) | Loss: 1.249609
Train Epoch: 8 | Batch Status: 32640/50000 (65%) | Loss: 1.429307
Train Epoch: 8 | Batch Status: 33280/50000 (66%) | Loss: 1.368118
Train Epoch: 8 | Batch Status: 33920/50000 (68%) | Loss: 1.115618
Train Epoch: 8 | Batch Status: 34560/50000 (69%) | Loss: 1.458943
Train Epoch: 8 | Batch Status: 35200/50000 (70%) | Loss: 1.270949
Train Epoch: 8 | Batch Status: 35840/50000 (72%) | Loss: 1.339936
Train Epoch: 8 | Batch Status: 36480/50000 (73%) | Loss: 1.477734
Train Epoch: 8 | Batch Status: 37120/50000 (74%) | Loss: 1.617551
Train Epoch: 8 | Batch Status: 37760/50000 (75%) | Loss: 1.485337
Train Epoch: 8 | Batch Status: 38400/50000 (77%) | Loss: 1.366454
Train Epoch: 8 | Batch Status: 39040/50000 (78%) | Loss: 1.341204
Train Epoch: 8 | Batch Status: 39680/50000 (79%) | Loss: 1.222241
Train Epoch: 8 | Batch Status: 40320/50000 (81%) | Loss: 1.216325
Train Epoch: 8 | Batch Status: 40960/50000 (82%) | Loss: 1.205903
Train Epoch: 8 | Batch Status: 41600/50000 (83%) | Loss: 1.344010
Train Epoch: 8 | Batch Status: 42240/50000 (84%) | Loss: 1.473661
Train Epoch: 8 | Batch Status: 42880/50000 (86%) | Loss: 1.409705
Train Epoch: 8 | Batch Status: 43520/50000 (87%) | Loss: 1.324390
Train Epoch: 8 | Batch Status: 44160/50000 (88%) | Loss: 1.307312
Train Epoch: 8 | Batch Status: 44800/50000 (90%) | Loss: 1.083642
Train Epoch: 8 | Batch Status: 45440/50000 (91%) | Loss: 1.248379
Train Epoch: 8 | Batch Status: 46080/50000 (92%) | Loss: 1.208865
Train Epoch: 8 | Batch Status: 46720/50000 (93%) | Loss: 1.214661
Train Epoch: 8 | Batch Status: 47360/50000 (95%) | Loss: 1.529662
Train Epoch: 8 | Batch Status: 48000/50000 (96%) | Loss: 1.288178
Train Epoch: 8 | Batch Status: 48640/50000 (97%) | Loss: 1.326149
Train Epoch: 8 | Batch Status: 49280/50000 (98%) | Loss: 1.110055
Train Epoch: 8 | Batch Status: 49920/50000 (100%) | Loss: 1.071278
Training time: 0m 10s
===========================
Test set: Average loss: 0.0224, Accuracy: 4926/10000 (49%)
Testing time: 0m 12s
Train Epoch: 9 | Batch Status: 0/50000 (0%) | Loss: 1.109779
Train Epoch: 9 | Batch Status: 640/50000 (1%) | Loss: 1.254291
Train Epoch: 9 | Batch Status: 1280/50000 (3%) | Loss: 1.056699
Train Epoch: 9 | Batch Status: 1920/50000 (4%) | Loss: 1.490433
Train Epoch: 9 | Batch Status: 2560/50000 (5%) | Loss: 1.517314
Train Epoch: 9 | Batch Status: 3200/50000 (6%) | Loss: 1.500928
Train Epoch: 9 | Batch Status: 3840/50000 (8%) | Loss: 1.335889
Train Epoch: 9 | Batch Status: 4480/50000 (9%) | Loss: 1.679691
Train Epoch: 9 | Batch Status: 5120/50000 (10%) | Loss: 1.309922
Train Epoch: 9 | Batch Status: 5760/50000 (12%) | Loss: 1.220353
Train Epoch: 9 | Batch Status: 6400/50000 (13%) | Loss: 1.342804
Train Epoch: 9 | Batch Status: 7040/50000 (14%) | Loss: 1.491447
Train Epoch: 9 | Batch Status: 7680/50000 (15%) | Loss: 1.161562
Train Epoch: 9 | Batch Status: 8320/50000 (17%) | Loss: 1.317864
Train Epoch: 9 | Batch Status: 8960/50000 (18%) | Loss: 1.301989
Train Epoch: 9 | Batch Status: 9600/50000 (19%) | Loss: 1.313591
Train Epoch: 9 | Batch Status: 10240/50000 (20%) | Loss: 1.237411
Train Epoch: 9 | Batch Status: 10880/50000 (22%) | Loss: 1.130649
Train Epoch: 9 | Batch Status: 11520/50000 (23%) | Loss: 1.012960
Train Epoch: 9 | Batch Status: 12160/50000 (24%) | Loss: 1.184511
Train Epoch: 9 | Batch Status: 12800/50000 (26%) | Loss: 1.166743
Train Epoch: 9 | Batch Status: 13440/50000 (27%) | Loss: 1.498342
Train Epoch: 9 | Batch Status: 14080/50000 (28%) | Loss: 1.208277
Train Epoch: 9 | Batch Status: 14720/50000 (29%) | Loss: 1.274428
Train Epoch: 9 | Batch Status: 15360/50000 (31%) | Loss: 1.295221
Train Epoch: 9 | Batch Status: 16000/50000 (32%) | Loss: 1.292833
Train Epoch: 9 | Batch Status: 16640/50000 (33%) | Loss: 1.575385
Train Epoch: 9 | Batch Status: 17280/50000 (35%) | Loss: 1.113047
Train Epoch: 9 | Batch Status: 17920/50000 (36%) | Loss: 1.474275
Train Epoch: 9 | Batch Status: 18560/50000 (37%) | Loss: 1.212395
Train Epoch: 9 | Batch Status: 19200/50000 (38%) | Loss: 1.352654
Train Epoch: 9 | Batch Status: 19840/50000 (40%) | Loss: 1.117064
Train Epoch: 9 | Batch Status: 20480/50000 (41%) | Loss: 1.570943
Train Epoch: 9 | Batch Status: 21120/50000 (42%) | Loss: 1.121025
Train Epoch: 9 | Batch Status: 21760/50000 (43%) | Loss: 1.452632
Train Epoch: 9 | Batch Status: 22400/50000 (45%) | Loss: 0.974265
Train Epoch: 9 | Batch Status: 23040/50000 (46%) | Loss: 1.447693
Train Epoch: 9 | Batch Status: 23680/50000 (47%) | Loss: 1.258151
Train Epoch: 9 | Batch Status: 24320/50000 (49%) | Loss: 1.102080
Train Epoch: 9 | Batch Status: 24960/50000 (50%) | Loss: 1.305913
Train Epoch: 9 | Batch Status: 25600/50000 (51%) | Loss: 1.369885
Train Epoch: 9 | Batch Status: 26240/50000 (52%) | Loss: 1.382820
Train Epoch: 9 | Batch Status: 26880/50000 (54%) | Loss: 1.244447
Train Epoch: 9 | Batch Status: 27520/50000 (55%) | Loss: 1.381393
Train Epoch: 9 | Batch Status: 28160/50000 (56%) | Loss: 1.506238
Train Epoch: 9 | Batch Status: 28800/50000 (58%) | Loss: 1.319295
Train Epoch: 9 | Batch Status: 29440/50000 (59%) | Loss: 1.466061
Train Epoch: 9 | Batch Status: 30080/50000 (60%) | Loss: 1.464418
Train Epoch: 9 | Batch Status: 30720/50000 (61%) | Loss: 1.032229
Train Epoch: 9 | Batch Status: 31360/50000 (63%) | Loss: 1.083711
Train Epoch: 9 | Batch Status: 32000/50000 (64%) | Loss: 1.270731
Train Epoch: 9 | Batch Status: 32640/50000 (65%) | Loss: 1.202548
Train Epoch: 9 | Batch Status: 33280/50000 (66%) | Loss: 1.348334
Train Epoch: 9 | Batch Status: 33920/50000 (68%) | Loss: 1.088614
Train Epoch: 9 | Batch Status: 34560/50000 (69%) | Loss: 1.209187
Train Epoch: 9 | Batch Status: 35200/50000 (70%) | Loss: 1.353218
Train Epoch: 9 | Batch Status: 35840/50000 (72%) | Loss: 1.154429
Train Epoch: 9 | Batch Status: 36480/50000 (73%) | Loss: 1.078067
Train Epoch: 9 | Batch Status: 37120/50000 (74%) | Loss: 1.370946
Train Epoch: 9 | Batch Status: 37760/50000 (75%) | Loss: 1.371708
Train Epoch: 9 | Batch Status: 38400/50000 (77%) | Loss: 1.409396
Train Epoch: 9 | Batch Status: 39040/50000 (78%) | Loss: 1.261564
Train Epoch: 9 | Batch Status: 39680/50000 (79%) | Loss: 1.247937
Train Epoch: 9 | Batch Status: 40320/50000 (81%) | Loss: 1.121768
Train Epoch: 9 | Batch Status: 40960/50000 (82%) | Loss: 1.310056
Train Epoch: 9 | Batch Status: 41600/50000 (83%) | Loss: 1.240692
Train Epoch: 9 | Batch Status: 42240/50000 (84%) | Loss: 1.262414
Train Epoch: 9 | Batch Status: 42880/50000 (86%) | Loss: 1.061443
Train Epoch: 9 | Batch Status: 43520/50000 (87%) | Loss: 1.167396
Train Epoch: 9 | Batch Status: 44160/50000 (88%) | Loss: 1.112253
Train Epoch: 9 | Batch Status: 44800/50000 (90%) | Loss: 1.046406
Train Epoch: 9 | Batch Status: 45440/50000 (91%) | Loss: 1.131242
Train Epoch: 9 | Batch Status: 46080/50000 (92%) | Loss: 1.231754
Train Epoch: 9 | Batch Status: 46720/50000 (93%) | Loss: 1.486003
Train Epoch: 9 | Batch Status: 47360/50000 (95%) | Loss: 1.123975
Train Epoch: 9 | Batch Status: 48000/50000 (96%) | Loss: 1.391704
Train Epoch: 9 | Batch Status: 48640/50000 (97%) | Loss: 1.265159
Train Epoch: 9 | Batch Status: 49280/50000 (98%) | Loss: 1.324484
Train Epoch: 9 | Batch Status: 49920/50000 (100%) | Loss: 1.171738
Training time: 0m 10s
===========================
Test set: Average loss: 0.0215, Accuracy: 5165/10000 (52%)
Testing time: 0m 12s
Total Time: 1m 40s
Model was trained on cuda!

Process finished with exit code 0
      
