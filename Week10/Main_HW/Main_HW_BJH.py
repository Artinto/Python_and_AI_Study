# https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import print_function
from torch import nn, optim, cuda
from torch.utils import data
from torchvision import datasets, transforms
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt


# Training settings
batch_size = 64
device = 'cuda' if cuda.is_available() else 'cpu' #cuda는 엔디비아사의 GPU기술 AMD는 OPENCL만 가능...ㅠ
print(f'Training MNIST Model on {device}\n{"=" * 44}')


# MNIST Dataset
#Train과 Test에 차이를 둔다.
train_dataset = datasets.CIFAR10(root='./mnist_data/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.CIFAR10(root='./mnist_data/',
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
        super(Net, self).__init__() #3 * 32 *32개가 기본제공 1024로 입력시 에러가 난다.
        self.l1 = nn.Linear(3*32*32, 1024)
        self.l2 = nn.Linear(1024, 512)
        self.l3 = nn.Linear(512, 170)
        self.l4 = nn.Linear(170, 100)
        self.l5 = nn.Linear(100, 10)

    def forward(self, x): #위에서 정의했던 과정을 사용하는 것
        x = x.view(-1, 3*32*32)  # Flatten the data (n, 1, 28, 28)-> (n, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)  #return이기에 따로 함수로 안해준다.


model = Net()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.015, momentum=0.9) #모멘텀은 전의 기울기를 약간 반영하는 식으로 계산 러닝레이트는 0.01 미만

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
      --------------------------------------------------------------------------------------
      
      
      
      
      
      
      
      
      C:\Anaconda3\python.exe "C:/Users/Jin Bong/Desktop/학교/Study/Cloud/09_2_softmax_mnist_BJH.py"
Training MNIST Model on cpu
============================================
Files already downloaded and verified
Train Epoch: 1 | Batch Status: 0/50000 (0%) | Loss: 2.294720
Train Epoch: 1 | Batch Status: 640/50000 (1%) | Loss: 2.310589
Train Epoch: 1 | Batch Status: 1280/50000 (3%) | Loss: 2.298670
Train Epoch: 1 | Batch Status: 1920/50000 (4%) | Loss: 2.306572
Train Epoch: 1 | Batch Status: 2560/50000 (5%) | Loss: 2.306335
Train Epoch: 1 | Batch Status: 3200/50000 (6%) | Loss: 2.277556
Train Epoch: 1 | Batch Status: 3840/50000 (8%) | Loss: 2.289602
Train Epoch: 1 | Batch Status: 4480/50000 (9%) | Loss: 2.286045
Train Epoch: 1 | Batch Status: 5120/50000 (10%) | Loss: 2.279018
Train Epoch: 1 | Batch Status: 5760/50000 (12%) | Loss: 2.247306
Train Epoch: 1 | Batch Status: 6400/50000 (13%) | Loss: 2.208623
Train Epoch: 1 | Batch Status: 7040/50000 (14%) | Loss: 2.207345
Train Epoch: 1 | Batch Status: 7680/50000 (15%) | Loss: 2.090355
Train Epoch: 1 | Batch Status: 8320/50000 (17%) | Loss: 2.106726
Train Epoch: 1 | Batch Status: 8960/50000 (18%) | Loss: 2.095600
Train Epoch: 1 | Batch Status: 9600/50000 (19%) | Loss: 2.064569
Train Epoch: 1 | Batch Status: 10240/50000 (20%) | Loss: 2.137781
Train Epoch: 1 | Batch Status: 10880/50000 (22%) | Loss: 2.147511
Train Epoch: 1 | Batch Status: 11520/50000 (23%) | Loss: 2.010552
Train Epoch: 1 | Batch Status: 12160/50000 (24%) | Loss: 2.004249
Train Epoch: 1 | Batch Status: 12800/50000 (26%) | Loss: 1.976193
Train Epoch: 1 | Batch Status: 13440/50000 (27%) | Loss: 2.168619
Train Epoch: 1 | Batch Status: 14080/50000 (28%) | Loss: 2.080812
Train Epoch: 1 | Batch Status: 14720/50000 (29%) | Loss: 1.990405
Train Epoch: 1 | Batch Status: 15360/50000 (31%) | Loss: 1.954829
Train Epoch: 1 | Batch Status: 16000/50000 (32%) | Loss: 2.153585
Train Epoch: 1 | Batch Status: 16640/50000 (33%) | Loss: 2.129220
Train Epoch: 1 | Batch Status: 17280/50000 (35%) | Loss: 1.869262
Train Epoch: 1 | Batch Status: 17920/50000 (36%) | Loss: 1.992992
Train Epoch: 1 | Batch Status: 18560/50000 (37%) | Loss: 1.998084
Train Epoch: 1 | Batch Status: 19200/50000 (38%) | Loss: 1.942132
Train Epoch: 1 | Batch Status: 19840/50000 (40%) | Loss: 1.849254
Train Epoch: 1 | Batch Status: 20480/50000 (41%) | Loss: 1.850201
Train Epoch: 1 | Batch Status: 21120/50000 (42%) | Loss: 1.888320
Train Epoch: 1 | Batch Status: 21760/50000 (43%) | Loss: 2.016863
Train Epoch: 1 | Batch Status: 22400/50000 (45%) | Loss: 1.886069
Train Epoch: 1 | Batch Status: 23040/50000 (46%) | Loss: 1.842823
Train Epoch: 1 | Batch Status: 23680/50000 (47%) | Loss: 1.875173
Train Epoch: 1 | Batch Status: 24320/50000 (49%) | Loss: 1.902688
Train Epoch: 1 | Batch Status: 24960/50000 (50%) | Loss: 1.888720
Train Epoch: 1 | Batch Status: 25600/50000 (51%) | Loss: 1.923099
Train Epoch: 1 | Batch Status: 26240/50000 (52%) | Loss: 1.879730
Train Epoch: 1 | Batch Status: 26880/50000 (54%) | Loss: 1.816950
Train Epoch: 1 | Batch Status: 27520/50000 (55%) | Loss: 1.764884
Train Epoch: 1 | Batch Status: 28160/50000 (56%) | Loss: 1.880213
Train Epoch: 1 | Batch Status: 28800/50000 (58%) | Loss: 1.787628
Train Epoch: 1 | Batch Status: 29440/50000 (59%) | Loss: 2.043888
Train Epoch: 1 | Batch Status: 30080/50000 (60%) | Loss: 1.873912
Train Epoch: 1 | Batch Status: 30720/50000 (61%) | Loss: 1.859969
Train Epoch: 1 | Batch Status: 31360/50000 (63%) | Loss: 1.934811
Train Epoch: 1 | Batch Status: 32000/50000 (64%) | Loss: 1.886775
Train Epoch: 1 | Batch Status: 32640/50000 (65%) | Loss: 1.803443
Train Epoch: 1 | Batch Status: 33280/50000 (66%) | Loss: 1.893142
Train Epoch: 1 | Batch Status: 33920/50000 (68%) | Loss: 1.894182
Train Epoch: 1 | Batch Status: 34560/50000 (69%) | Loss: 1.874029
Train Epoch: 1 | Batch Status: 35200/50000 (70%) | Loss: 1.956075
Train Epoch: 1 | Batch Status: 35840/50000 (72%) | Loss: 1.944267
Train Epoch: 1 | Batch Status: 36480/50000 (73%) | Loss: 1.864925
Train Epoch: 1 | Batch Status: 37120/50000 (74%) | Loss: 1.880536
Train Epoch: 1 | Batch Status: 37760/50000 (75%) | Loss: 1.920218
Train Epoch: 1 | Batch Status: 38400/50000 (77%) | Loss: 1.867233
Train Epoch: 1 | Batch Status: 39040/50000 (78%) | Loss: 1.997179
Train Epoch: 1 | Batch Status: 39680/50000 (79%) | Loss: 1.802085
Train Epoch: 1 | Batch Status: 40320/50000 (81%) | Loss: 1.916028
Train Epoch: 1 | Batch Status: 40960/50000 (82%) | Loss: 1.680501
Train Epoch: 1 | Batch Status: 41600/50000 (83%) | Loss: 1.665536
Train Epoch: 1 | Batch Status: 42240/50000 (84%) | Loss: 1.758056
Train Epoch: 1 | Batch Status: 42880/50000 (86%) | Loss: 1.831136
Train Epoch: 1 | Batch Status: 43520/50000 (87%) | Loss: 1.829075
Train Epoch: 1 | Batch Status: 44160/50000 (88%) | Loss: 1.840235
Train Epoch: 1 | Batch Status: 44800/50000 (90%) | Loss: 1.946551
Train Epoch: 1 | Batch Status: 45440/50000 (91%) | Loss: 1.882332
Train Epoch: 1 | Batch Status: 46080/50000 (92%) | Loss: 1.906332
Train Epoch: 1 | Batch Status: 46720/50000 (93%) | Loss: 1.909388
Train Epoch: 1 | Batch Status: 47360/50000 (95%) | Loss: 1.944268
Train Epoch: 1 | Batch Status: 48000/50000 (96%) | Loss: 1.894588
Train Epoch: 1 | Batch Status: 48640/50000 (97%) | Loss: 1.977090
Train Epoch: 1 | Batch Status: 49280/50000 (98%) | Loss: 1.662683
Train Epoch: 1 | Batch Status: 49920/50000 (100%) | Loss: 1.715642
Training time: 0m 21s
===========================
Test set: Average loss: 0.0278, Accuracy: 3550/10000 (36%)
Testing time: 0m 22s
Train Epoch: 2 | Batch Status: 0/50000 (0%) | Loss: 1.732647
Train Epoch: 2 | Batch Status: 640/50000 (1%) | Loss: 1.669933
Train Epoch: 2 | Batch Status: 1280/50000 (3%) | Loss: 2.047984
Train Epoch: 2 | Batch Status: 1920/50000 (4%) | Loss: 2.024454
Train Epoch: 2 | Batch Status: 2560/50000 (5%) | Loss: 1.731786
Train Epoch: 2 | Batch Status: 3200/50000 (6%) | Loss: 1.759544
Train Epoch: 2 | Batch Status: 3840/50000 (8%) | Loss: 1.919514
Train Epoch: 2 | Batch Status: 4480/50000 (9%) | Loss: 1.669948
Train Epoch: 2 | Batch Status: 5120/50000 (10%) | Loss: 1.861453
Train Epoch: 2 | Batch Status: 5760/50000 (12%) | Loss: 1.949421
Train Epoch: 2 | Batch Status: 6400/50000 (13%) | Loss: 1.886711
Train Epoch: 2 | Batch Status: 7040/50000 (14%) | Loss: 1.729446
Train Epoch: 2 | Batch Status: 7680/50000 (15%) | Loss: 1.841160
Train Epoch: 2 | Batch Status: 8320/50000 (17%) | Loss: 1.855770
Train Epoch: 2 | Batch Status: 8960/50000 (18%) | Loss: 1.642552
Train Epoch: 2 | Batch Status: 9600/50000 (19%) | Loss: 1.940357
Train Epoch: 2 | Batch Status: 10240/50000 (20%) | Loss: 1.842720
Train Epoch: 2 | Batch Status: 10880/50000 (22%) | Loss: 1.697537
Train Epoch: 2 | Batch Status: 11520/50000 (23%) | Loss: 1.731786
Train Epoch: 2 | Batch Status: 12160/50000 (24%) | Loss: 1.886755
Train Epoch: 2 | Batch Status: 12800/50000 (26%) | Loss: 1.762982
Train Epoch: 2 | Batch Status: 13440/50000 (27%) | Loss: 1.627829
Train Epoch: 2 | Batch Status: 14080/50000 (28%) | Loss: 1.773056
Train Epoch: 2 | Batch Status: 14720/50000 (29%) | Loss: 1.894534
Train Epoch: 2 | Batch Status: 15360/50000 (31%) | Loss: 1.611132
Train Epoch: 2 | Batch Status: 16000/50000 (32%) | Loss: 1.698400
Train Epoch: 2 | Batch Status: 16640/50000 (33%) | Loss: 1.970069
Train Epoch: 2 | Batch Status: 17280/50000 (35%) | Loss: 1.921826
Train Epoch: 2 | Batch Status: 17920/50000 (36%) | Loss: 1.685431
Train Epoch: 2 | Batch Status: 18560/50000 (37%) | Loss: 1.719024
Train Epoch: 2 | Batch Status: 19200/50000 (38%) | Loss: 1.857284
Train Epoch: 2 | Batch Status: 19840/50000 (40%) | Loss: 1.900382
Train Epoch: 2 | Batch Status: 20480/50000 (41%) | Loss: 1.796685
Train Epoch: 2 | Batch Status: 21120/50000 (42%) | Loss: 1.665246
Train Epoch: 2 | Batch Status: 21760/50000 (43%) | Loss: 1.766904
Train Epoch: 2 | Batch Status: 22400/50000 (45%) | Loss: 1.725054
Train Epoch: 2 | Batch Status: 23040/50000 (46%) | Loss: 1.596098
Train Epoch: 2 | Batch Status: 23680/50000 (47%) | Loss: 1.852398
Train Epoch: 2 | Batch Status: 24320/50000 (49%) | Loss: 1.807849
Train Epoch: 2 | Batch Status: 24960/50000 (50%) | Loss: 1.778484
Train Epoch: 2 | Batch Status: 25600/50000 (51%) | Loss: 1.577309
Train Epoch: 2 | Batch Status: 26240/50000 (52%) | Loss: 1.704119
Train Epoch: 2 | Batch Status: 26880/50000 (54%) | Loss: 1.655421
Train Epoch: 2 | Batch Status: 27520/50000 (55%) | Loss: 2.054751
Train Epoch: 2 | Batch Status: 28160/50000 (56%) | Loss: 1.652394
Train Epoch: 2 | Batch Status: 28800/50000 (58%) | Loss: 1.760873
Train Epoch: 2 | Batch Status: 29440/50000 (59%) | Loss: 1.649071
Train Epoch: 2 | Batch Status: 30080/50000 (60%) | Loss: 1.614414
Train Epoch: 2 | Batch Status: 30720/50000 (61%) | Loss: 1.867179
Train Epoch: 2 | Batch Status: 31360/50000 (63%) | Loss: 1.708878
Train Epoch: 2 | Batch Status: 32000/50000 (64%) | Loss: 1.540263
Train Epoch: 2 | Batch Status: 32640/50000 (65%) | Loss: 1.538354
Train Epoch: 2 | Batch Status: 33280/50000 (66%) | Loss: 1.673740
Train Epoch: 2 | Batch Status: 33920/50000 (68%) | Loss: 1.620951
Train Epoch: 2 | Batch Status: 34560/50000 (69%) | Loss: 1.933069
Train Epoch: 2 | Batch Status: 35200/50000 (70%) | Loss: 1.583251
Train Epoch: 2 | Batch Status: 35840/50000 (72%) | Loss: 1.634879
Train Epoch: 2 | Batch Status: 36480/50000 (73%) | Loss: 1.720271
Train Epoch: 2 | Batch Status: 37120/50000 (74%) | Loss: 1.619835
Train Epoch: 2 | Batch Status: 37760/50000 (75%) | Loss: 1.836005
Train Epoch: 2 | Batch Status: 38400/50000 (77%) | Loss: 1.611380
Train Epoch: 2 | Batch Status: 39040/50000 (78%) | Loss: 1.717078
Train Epoch: 2 | Batch Status: 39680/50000 (79%) | Loss: 1.660252
Train Epoch: 2 | Batch Status: 40320/50000 (81%) | Loss: 1.598109
Train Epoch: 2 | Batch Status: 40960/50000 (82%) | Loss: 1.707219
Train Epoch: 2 | Batch Status: 41600/50000 (83%) | Loss: 1.908147
Train Epoch: 2 | Batch Status: 42240/50000 (84%) | Loss: 1.572028
Train Epoch: 2 | Batch Status: 42880/50000 (86%) | Loss: 1.681852
Train Epoch: 2 | Batch Status: 43520/50000 (87%) | Loss: 1.530998
Train Epoch: 2 | Batch Status: 44160/50000 (88%) | Loss: 1.761120
Train Epoch: 2 | Batch Status: 44800/50000 (90%) | Loss: 1.686549
Train Epoch: 2 | Batch Status: 45440/50000 (91%) | Loss: 1.582900
Train Epoch: 2 | Batch Status: 46080/50000 (92%) | Loss: 1.620027
Train Epoch: 2 | Batch Status: 46720/50000 (93%) | Loss: 1.707231
Train Epoch: 2 | Batch Status: 47360/50000 (95%) | Loss: 1.571294
Train Epoch: 2 | Batch Status: 48000/50000 (96%) | Loss: 1.660057
Train Epoch: 2 | Batch Status: 48640/50000 (97%) | Loss: 1.883730
Train Epoch: 2 | Batch Status: 49280/50000 (98%) | Loss: 1.603104
Train Epoch: 2 | Batch Status: 49920/50000 (100%) | Loss: 1.576608
Training time: 0m 23s
===========================
Test set: Average loss: 0.0266, Accuracy: 4066/10000 (41%)
Testing time: 0m 25s
Train Epoch: 3 | Batch Status: 0/50000 (0%) | Loss: 1.708360
Train Epoch: 3 | Batch Status: 640/50000 (1%) | Loss: 1.605322
Train Epoch: 3 | Batch Status: 1280/50000 (3%) | Loss: 1.673822
Train Epoch: 3 | Batch Status: 1920/50000 (4%) | Loss: 1.540688
Train Epoch: 3 | Batch Status: 2560/50000 (5%) | Loss: 1.974648
Train Epoch: 3 | Batch Status: 3200/50000 (6%) | Loss: 1.769737
Train Epoch: 3 | Batch Status: 3840/50000 (8%) | Loss: 1.626455
Train Epoch: 3 | Batch Status: 4480/50000 (9%) | Loss: 1.651901
Train Epoch: 3 | Batch Status: 5120/50000 (10%) | Loss: 1.971705
Train Epoch: 3 | Batch Status: 5760/50000 (12%) | Loss: 1.622195
Train Epoch: 3 | Batch Status: 6400/50000 (13%) | Loss: 1.516009
Train Epoch: 3 | Batch Status: 7040/50000 (14%) | Loss: 1.592936
Train Epoch: 3 | Batch Status: 7680/50000 (15%) | Loss: 1.574432
Train Epoch: 3 | Batch Status: 8320/50000 (17%) | Loss: 1.610836
Train Epoch: 3 | Batch Status: 8960/50000 (18%) | Loss: 1.704980
Train Epoch: 3 | Batch Status: 9600/50000 (19%) | Loss: 1.661742
Train Epoch: 3 | Batch Status: 10240/50000 (20%) | Loss: 1.700425
Train Epoch: 3 | Batch Status: 10880/50000 (22%) | Loss: 1.813928
Train Epoch: 3 | Batch Status: 11520/50000 (23%) | Loss: 1.673800
Train Epoch: 3 | Batch Status: 12160/50000 (24%) | Loss: 1.556790
Train Epoch: 3 | Batch Status: 12800/50000 (26%) | Loss: 1.436486
Train Epoch: 3 | Batch Status: 13440/50000 (27%) | Loss: 1.869761
Train Epoch: 3 | Batch Status: 14080/50000 (28%) | Loss: 1.680232
Train Epoch: 3 | Batch Status: 14720/50000 (29%) | Loss: 2.085022
Train Epoch: 3 | Batch Status: 15360/50000 (31%) | Loss: 1.604817
Train Epoch: 3 | Batch Status: 16000/50000 (32%) | Loss: 1.520944
Train Epoch: 3 | Batch Status: 16640/50000 (33%) | Loss: 1.742542
Train Epoch: 3 | Batch Status: 17280/50000 (35%) | Loss: 1.730434
Train Epoch: 3 | Batch Status: 17920/50000 (36%) | Loss: 1.672362
Train Epoch: 3 | Batch Status: 18560/50000 (37%) | Loss: 1.569695
Train Epoch: 3 | Batch Status: 19200/50000 (38%) | Loss: 1.642187
Train Epoch: 3 | Batch Status: 19840/50000 (40%) | Loss: 1.792102
Train Epoch: 3 | Batch Status: 20480/50000 (41%) | Loss: 1.670204
Train Epoch: 3 | Batch Status: 21120/50000 (42%) | Loss: 1.747258
Train Epoch: 3 | Batch Status: 21760/50000 (43%) | Loss: 1.588262
Train Epoch: 3 | Batch Status: 22400/50000 (45%) | Loss: 1.656209
Train Epoch: 3 | Batch Status: 23040/50000 (46%) | Loss: 1.613388
Train Epoch: 3 | Batch Status: 23680/50000 (47%) | Loss: 1.869707
Train Epoch: 3 | Batch Status: 24320/50000 (49%) | Loss: 1.701418
Train Epoch: 3 | Batch Status: 24960/50000 (50%) | Loss: 1.709445
Train Epoch: 3 | Batch Status: 25600/50000 (51%) | Loss: 1.464629
Train Epoch: 3 | Batch Status: 26240/50000 (52%) | Loss: 1.602202
Train Epoch: 3 | Batch Status: 26880/50000 (54%) | Loss: 1.548079
Train Epoch: 3 | Batch Status: 27520/50000 (55%) | Loss: 1.665904
Train Epoch: 3 | Batch Status: 28160/50000 (56%) | Loss: 1.735765
Train Epoch: 3 | Batch Status: 28800/50000 (58%) | Loss: 1.582714
Train Epoch: 3 | Batch Status: 29440/50000 (59%) | Loss: 1.614335
Train Epoch: 3 | Batch Status: 30080/50000 (60%) | Loss: 1.800326
Train Epoch: 3 | Batch Status: 30720/50000 (61%) | Loss: 1.640731
Train Epoch: 3 | Batch Status: 31360/50000 (63%) | Loss: 1.825334
Train Epoch: 3 | Batch Status: 32000/50000 (64%) | Loss: 1.649527
Train Epoch: 3 | Batch Status: 32640/50000 (65%) | Loss: 1.691757
Train Epoch: 3 | Batch Status: 33280/50000 (66%) | Loss: 1.687913
Train Epoch: 3 | Batch Status: 33920/50000 (68%) | Loss: 1.637301
Train Epoch: 3 | Batch Status: 34560/50000 (69%) | Loss: 1.397874
Train Epoch: 3 | Batch Status: 35200/50000 (70%) | Loss: 1.503832
Train Epoch: 3 | Batch Status: 35840/50000 (72%) | Loss: 1.747970
Train Epoch: 3 | Batch Status: 36480/50000 (73%) | Loss: 1.634416
Train Epoch: 3 | Batch Status: 37120/50000 (74%) | Loss: 1.472360
Train Epoch: 3 | Batch Status: 37760/50000 (75%) | Loss: 1.584099
Train Epoch: 3 | Batch Status: 38400/50000 (77%) | Loss: 1.604701
Train Epoch: 3 | Batch Status: 39040/50000 (78%) | Loss: 1.437365
Train Epoch: 3 | Batch Status: 39680/50000 (79%) | Loss: 1.558171
Train Epoch: 3 | Batch Status: 40320/50000 (81%) | Loss: 1.759001
Train Epoch: 3 | Batch Status: 40960/50000 (82%) | Loss: 1.779299
Train Epoch: 3 | Batch Status: 41600/50000 (83%) | Loss: 1.563284
Train Epoch: 3 | Batch Status: 42240/50000 (84%) | Loss: 1.694250
Train Epoch: 3 | Batch Status: 42880/50000 (86%) | Loss: 1.467108
Train Epoch: 3 | Batch Status: 43520/50000 (87%) | Loss: 1.788490
Train Epoch: 3 | Batch Status: 44160/50000 (88%) | Loss: 1.679710
Train Epoch: 3 | Batch Status: 44800/50000 (90%) | Loss: 1.602079
Train Epoch: 3 | Batch Status: 45440/50000 (91%) | Loss: 1.442120
Train Epoch: 3 | Batch Status: 46080/50000 (92%) | Loss: 1.521020
Train Epoch: 3 | Batch Status: 46720/50000 (93%) | Loss: 1.703879
Train Epoch: 3 | Batch Status: 47360/50000 (95%) | Loss: 1.576831
Train Epoch: 3 | Batch Status: 48000/50000 (96%) | Loss: 1.524076
Train Epoch: 3 | Batch Status: 48640/50000 (97%) | Loss: 1.900484
Train Epoch: 3 | Batch Status: 49280/50000 (98%) | Loss: 1.570723
Train Epoch: 3 | Batch Status: 49920/50000 (100%) | Loss: 1.500544
Training time: 0m 23s
===========================
Test set: Average loss: 0.0273, Accuracy: 3948/10000 (39%)
Testing time: 0m 26s
Train Epoch: 4 | Batch Status: 0/50000 (0%) | Loss: 1.974295
Train Epoch: 4 | Batch Status: 640/50000 (1%) | Loss: 1.662036
Train Epoch: 4 | Batch Status: 1280/50000 (3%) | Loss: 1.450895
Train Epoch: 4 | Batch Status: 1920/50000 (4%) | Loss: 1.518304
Train Epoch: 4 | Batch Status: 2560/50000 (5%) | Loss: 1.542878
Train Epoch: 4 | Batch Status: 3200/50000 (6%) | Loss: 1.555309
Train Epoch: 4 | Batch Status: 3840/50000 (8%) | Loss: 1.789348
Train Epoch: 4 | Batch Status: 4480/50000 (9%) | Loss: 1.652307
Train Epoch: 4 | Batch Status: 5120/50000 (10%) | Loss: 1.572114
Train Epoch: 4 | Batch Status: 5760/50000 (12%) | Loss: 1.666828
Train Epoch: 4 | Batch Status: 6400/50000 (13%) | Loss: 1.684520
Train Epoch: 4 | Batch Status: 7040/50000 (14%) | Loss: 1.679905
Train Epoch: 4 | Batch Status: 7680/50000 (15%) | Loss: 1.479596
Train Epoch: 4 | Batch Status: 8320/50000 (17%) | Loss: 1.432435
Train Epoch: 4 | Batch Status: 8960/50000 (18%) | Loss: 1.715124
Train Epoch: 4 | Batch Status: 9600/50000 (19%) | Loss: 1.625786
Train Epoch: 4 | Batch Status: 10240/50000 (20%) | Loss: 1.720493
Train Epoch: 4 | Batch Status: 10880/50000 (22%) | Loss: 1.560793
Train Epoch: 4 | Batch Status: 11520/50000 (23%) | Loss: 1.762085
Train Epoch: 4 | Batch Status: 12160/50000 (24%) | Loss: 1.626080
Train Epoch: 4 | Batch Status: 12800/50000 (26%) | Loss: 1.684669
Train Epoch: 4 | Batch Status: 13440/50000 (27%) | Loss: 1.420594
Train Epoch: 4 | Batch Status: 14080/50000 (28%) | Loss: 1.844216
Train Epoch: 4 | Batch Status: 14720/50000 (29%) | Loss: 1.476809
Train Epoch: 4 | Batch Status: 15360/50000 (31%) | Loss: 1.654251
Train Epoch: 4 | Batch Status: 16000/50000 (32%) | Loss: 1.780814
Train Epoch: 4 | Batch Status: 16640/50000 (33%) | Loss: 1.370255
Train Epoch: 4 | Batch Status: 17280/50000 (35%) | Loss: 1.547278
Train Epoch: 4 | Batch Status: 17920/50000 (36%) | Loss: 1.418679
Train Epoch: 4 | Batch Status: 18560/50000 (37%) | Loss: 1.553741
Train Epoch: 4 | Batch Status: 19200/50000 (38%) | Loss: 1.892472
Train Epoch: 4 | Batch Status: 19840/50000 (40%) | Loss: 1.708234
Train Epoch: 4 | Batch Status: 20480/50000 (41%) | Loss: 1.446574
Train Epoch: 4 | Batch Status: 21120/50000 (42%) | Loss: 1.655409
Train Epoch: 4 | Batch Status: 21760/50000 (43%) | Loss: 1.465330
Train Epoch: 4 | Batch Status: 22400/50000 (45%) | Loss: 1.688249
Train Epoch: 4 | Batch Status: 23040/50000 (46%) | Loss: 1.513274
Train Epoch: 4 | Batch Status: 23680/50000 (47%) | Loss: 1.570461
Train Epoch: 4 | Batch Status: 24320/50000 (49%) | Loss: 1.540738
Train Epoch: 4 | Batch Status: 24960/50000 (50%) | Loss: 1.478665
Train Epoch: 4 | Batch Status: 25600/50000 (51%) | Loss: 1.654648
Train Epoch: 4 | Batch Status: 26240/50000 (52%) | Loss: 1.529037
Train Epoch: 4 | Batch Status: 26880/50000 (54%) | Loss: 1.657746
Train Epoch: 4 | Batch Status: 27520/50000 (55%) | Loss: 1.630383
Train Epoch: 4 | Batch Status: 28160/50000 (56%) | Loss: 1.493162
Train Epoch: 4 | Batch Status: 28800/50000 (58%) | Loss: 1.634083
Train Epoch: 4 | Batch Status: 29440/50000 (59%) | Loss: 1.459696
Train Epoch: 4 | Batch Status: 30080/50000 (60%) | Loss: 1.484221
Train Epoch: 4 | Batch Status: 30720/50000 (61%) | Loss: 1.529975
Train Epoch: 4 | Batch Status: 31360/50000 (63%) | Loss: 1.624259
Train Epoch: 4 | Batch Status: 32000/50000 (64%) | Loss: 1.558935
Train Epoch: 4 | Batch Status: 32640/50000 (65%) | Loss: 1.419287
Train Epoch: 4 | Batch Status: 33280/50000 (66%) | Loss: 1.488079
Train Epoch: 4 | Batch Status: 33920/50000 (68%) | Loss: 1.781144
Train Epoch: 4 | Batch Status: 34560/50000 (69%) | Loss: 1.600626
Train Epoch: 4 | Batch Status: 35200/50000 (70%) | Loss: 1.695827
Train Epoch: 4 | Batch Status: 35840/50000 (72%) | Loss: 1.532374
Train Epoch: 4 | Batch Status: 36480/50000 (73%) | Loss: 1.766261
Train Epoch: 4 | Batch Status: 37120/50000 (74%) | Loss: 1.503371
Train Epoch: 4 | Batch Status: 37760/50000 (75%) | Loss: 1.524509
Train Epoch: 4 | Batch Status: 38400/50000 (77%) | Loss: 1.650707
Train Epoch: 4 | Batch Status: 39040/50000 (78%) | Loss: 1.833784
Train Epoch: 4 | Batch Status: 39680/50000 (79%) | Loss: 1.823336
Train Epoch: 4 | Batch Status: 40320/50000 (81%) | Loss: 1.664739
Train Epoch: 4 | Batch Status: 40960/50000 (82%) | Loss: 1.599247
Train Epoch: 4 | Batch Status: 41600/50000 (83%) | Loss: 1.663875
Train Epoch: 4 | Batch Status: 42240/50000 (84%) | Loss: 1.585717
Train Epoch: 4 | Batch Status: 42880/50000 (86%) | Loss: 1.345592
Train Epoch: 4 | Batch Status: 43520/50000 (87%) | Loss: 1.569809
Train Epoch: 4 | Batch Status: 44160/50000 (88%) | Loss: 1.712976
Train Epoch: 4 | Batch Status: 44800/50000 (90%) | Loss: 1.498466
Train Epoch: 4 | Batch Status: 45440/50000 (91%) | Loss: 1.509658
Train Epoch: 4 | Batch Status: 46080/50000 (92%) | Loss: 1.550946
Train Epoch: 4 | Batch Status: 46720/50000 (93%) | Loss: 1.731766
Train Epoch: 4 | Batch Status: 47360/50000 (95%) | Loss: 1.541325
Train Epoch: 4 | Batch Status: 48000/50000 (96%) | Loss: 1.617829
Train Epoch: 4 | Batch Status: 48640/50000 (97%) | Loss: 1.618709
Train Epoch: 4 | Batch Status: 49280/50000 (98%) | Loss: 1.553988
Train Epoch: 4 | Batch Status: 49920/50000 (100%) | Loss: 1.544492
Training time: 0m 23s
===========================
Test set: Average loss: 0.0249, Accuracy: 4470/10000 (45%)
Testing time: 0m 26s
Train Epoch: 5 | Batch Status: 0/50000 (0%) | Loss: 1.544061
Train Epoch: 5 | Batch Status: 640/50000 (1%) | Loss: 1.610041
Train Epoch: 5 | Batch Status: 1280/50000 (3%) | Loss: 1.484021
Train Epoch: 5 | Batch Status: 1920/50000 (4%) | Loss: 1.613037
Train Epoch: 5 | Batch Status: 2560/50000 (5%) | Loss: 1.416401
Train Epoch: 5 | Batch Status: 3200/50000 (6%) | Loss: 1.661958
Train Epoch: 5 | Batch Status: 3840/50000 (8%) | Loss: 1.569971
Train Epoch: 5 | Batch Status: 4480/50000 (9%) | Loss: 1.403124
Train Epoch: 5 | Batch Status: 5120/50000 (10%) | Loss: 1.650740
Train Epoch: 5 | Batch Status: 5760/50000 (12%) | Loss: 1.280188
Train Epoch: 5 | Batch Status: 6400/50000 (13%) | Loss: 1.478047
Train Epoch: 5 | Batch Status: 7040/50000 (14%) | Loss: 1.690664
Train Epoch: 5 | Batch Status: 7680/50000 (15%) | Loss: 1.690328
Train Epoch: 5 | Batch Status: 8320/50000 (17%) | Loss: 1.417721
Train Epoch: 5 | Batch Status: 8960/50000 (18%) | Loss: 1.500956
Train Epoch: 5 | Batch Status: 9600/50000 (19%) | Loss: 1.857237
Train Epoch: 5 | Batch Status: 10240/50000 (20%) | Loss: 1.621723
Train Epoch: 5 | Batch Status: 10880/50000 (22%) | Loss: 1.668469
Train Epoch: 5 | Batch Status: 11520/50000 (23%) | Loss: 1.454175
Train Epoch: 5 | Batch Status: 12160/50000 (24%) | Loss: 1.451094
Train Epoch: 5 | Batch Status: 12800/50000 (26%) | Loss: 1.744591
Train Epoch: 5 | Batch Status: 13440/50000 (27%) | Loss: 1.416688
Train Epoch: 5 | Batch Status: 14080/50000 (28%) | Loss: 1.407912
Train Epoch: 5 | Batch Status: 14720/50000 (29%) | Loss: 1.261400
Train Epoch: 5 | Batch Status: 15360/50000 (31%) | Loss: 1.624209
Train Epoch: 5 | Batch Status: 16000/50000 (32%) | Loss: 1.555613
Train Epoch: 5 | Batch Status: 16640/50000 (33%) | Loss: 1.649299
Train Epoch: 5 | Batch Status: 17280/50000 (35%) | Loss: 1.518186
Train Epoch: 5 | Batch Status: 17920/50000 (36%) | Loss: 1.447596
Train Epoch: 5 | Batch Status: 18560/50000 (37%) | Loss: 1.303017
Train Epoch: 5 | Batch Status: 19200/50000 (38%) | Loss: 1.329084
Train Epoch: 5 | Batch Status: 19840/50000 (40%) | Loss: 1.595870
Train Epoch: 5 | Batch Status: 20480/50000 (41%) | Loss: 1.478502
Train Epoch: 5 | Batch Status: 21120/50000 (42%) | Loss: 1.590292
Train Epoch: 5 | Batch Status: 21760/50000 (43%) | Loss: 1.341445
Train Epoch: 5 | Batch Status: 22400/50000 (45%) | Loss: 1.770734
Train Epoch: 5 | Batch Status: 23040/50000 (46%) | Loss: 1.486346
Train Epoch: 5 | Batch Status: 23680/50000 (47%) | Loss: 1.659997
Train Epoch: 5 | Batch Status: 24320/50000 (49%) | Loss: 1.440441
Train Epoch: 5 | Batch Status: 24960/50000 (50%) | Loss: 1.543242
Train Epoch: 5 | Batch Status: 25600/50000 (51%) | Loss: 1.629474
Train Epoch: 5 | Batch Status: 26240/50000 (52%) | Loss: 1.768873
Train Epoch: 5 | Batch Status: 26880/50000 (54%) | Loss: 1.609193
Train Epoch: 5 | Batch Status: 27520/50000 (55%) | Loss: 1.816892
Train Epoch: 5 | Batch Status: 28160/50000 (56%) | Loss: 1.400144
Train Epoch: 5 | Batch Status: 28800/50000 (58%) | Loss: 1.523072
Train Epoch: 5 | Batch Status: 29440/50000 (59%) | Loss: 1.724841
Train Epoch: 5 | Batch Status: 30080/50000 (60%) | Loss: 1.597434
Train Epoch: 5 | Batch Status: 30720/50000 (61%) | Loss: 1.420673
Train Epoch: 5 | Batch Status: 31360/50000 (63%) | Loss: 1.492570
Train Epoch: 5 | Batch Status: 32000/50000 (64%) | Loss: 1.576845
Train Epoch: 5 | Batch Status: 32640/50000 (65%) | Loss: 1.585531
Train Epoch: 5 | Batch Status: 33280/50000 (66%) | Loss: 1.254488
Train Epoch: 5 | Batch Status: 33920/50000 (68%) | Loss: 1.406847
Train Epoch: 5 | Batch Status: 34560/50000 (69%) | Loss: 1.488429
Train Epoch: 5 | Batch Status: 35200/50000 (70%) | Loss: 1.797631
Train Epoch: 5 | Batch Status: 35840/50000 (72%) | Loss: 1.492435
Train Epoch: 5 | Batch Status: 36480/50000 (73%) | Loss: 1.328989
Train Epoch: 5 | Batch Status: 37120/50000 (74%) | Loss: 1.596313
Train Epoch: 5 | Batch Status: 37760/50000 (75%) | Loss: 1.633512
Train Epoch: 5 | Batch Status: 38400/50000 (77%) | Loss: 1.495844
Train Epoch: 5 | Batch Status: 39040/50000 (78%) | Loss: 1.401365
Train Epoch: 5 | Batch Status: 39680/50000 (79%) | Loss: 1.427058
Train Epoch: 5 | Batch Status: 40320/50000 (81%) | Loss: 1.472452
Train Epoch: 5 | Batch Status: 40960/50000 (82%) | Loss: 1.407396
Train Epoch: 5 | Batch Status: 41600/50000 (83%) | Loss: 1.528660
Train Epoch: 5 | Batch Status: 42240/50000 (84%) | Loss: 1.472291
Train Epoch: 5 | Batch Status: 42880/50000 (86%) | Loss: 1.495522
Train Epoch: 5 | Batch Status: 43520/50000 (87%) | Loss: 1.541109
Train Epoch: 5 | Batch Status: 44160/50000 (88%) | Loss: 1.666844
Train Epoch: 5 | Batch Status: 44800/50000 (90%) | Loss: 1.460485
Train Epoch: 5 | Batch Status: 45440/50000 (91%) | Loss: 1.456152
Train Epoch: 5 | Batch Status: 46080/50000 (92%) | Loss: 1.693685
Train Epoch: 5 | Batch Status: 46720/50000 (93%) | Loss: 1.530150
Train Epoch: 5 | Batch Status: 47360/50000 (95%) | Loss: 1.541986
Train Epoch: 5 | Batch Status: 48000/50000 (96%) | Loss: 1.471489
Train Epoch: 5 | Batch Status: 48640/50000 (97%) | Loss: 1.461477
Train Epoch: 5 | Batch Status: 49280/50000 (98%) | Loss: 1.375965
Train Epoch: 5 | Batch Status: 49920/50000 (100%) | Loss: 1.472002
Training time: 0m 23s
===========================
Test set: Average loss: 0.0234, Accuracy: 4640/10000 (46%)
Testing time: 0m 25s
Train Epoch: 6 | Batch Status: 0/50000 (0%) | Loss: 1.537952
Train Epoch: 6 | Batch Status: 640/50000 (1%) | Loss: 1.724898
Train Epoch: 6 | Batch Status: 1280/50000 (3%) | Loss: 1.396621
Train Epoch: 6 | Batch Status: 1920/50000 (4%) | Loss: 1.520592
Train Epoch: 6 | Batch Status: 2560/50000 (5%) | Loss: 1.565313
Train Epoch: 6 | Batch Status: 3200/50000 (6%) | Loss: 1.371417
Train Epoch: 6 | Batch Status: 3840/50000 (8%) | Loss: 1.392436
Train Epoch: 6 | Batch Status: 4480/50000 (9%) | Loss: 1.354016
Train Epoch: 6 | Batch Status: 5120/50000 (10%) | Loss: 1.273930
Train Epoch: 6 | Batch Status: 5760/50000 (12%) | Loss: 1.494805
Train Epoch: 6 | Batch Status: 6400/50000 (13%) | Loss: 1.387251
Train Epoch: 6 | Batch Status: 7040/50000 (14%) | Loss: 1.313850
Train Epoch: 6 | Batch Status: 7680/50000 (15%) | Loss: 1.540847
Train Epoch: 6 | Batch Status: 8320/50000 (17%) | Loss: 1.444751
Train Epoch: 6 | Batch Status: 8960/50000 (18%) | Loss: 1.534668
Train Epoch: 6 | Batch Status: 9600/50000 (19%) | Loss: 1.667037
Train Epoch: 6 | Batch Status: 10240/50000 (20%) | Loss: 1.191703
Train Epoch: 6 | Batch Status: 10880/50000 (22%) | Loss: 1.340937
Train Epoch: 6 | Batch Status: 11520/50000 (23%) | Loss: 1.514124
Train Epoch: 6 | Batch Status: 12160/50000 (24%) | Loss: 1.481810
Train Epoch: 6 | Batch Status: 12800/50000 (26%) | Loss: 1.491032
Train Epoch: 6 | Batch Status: 13440/50000 (27%) | Loss: 1.264978
Train Epoch: 6 | Batch Status: 14080/50000 (28%) | Loss: 1.517921
Train Epoch: 6 | Batch Status: 14720/50000 (29%) | Loss: 1.650287
Train Epoch: 6 | Batch Status: 15360/50000 (31%) | Loss: 1.377058
Train Epoch: 6 | Batch Status: 16000/50000 (32%) | Loss: 1.569858
Train Epoch: 6 | Batch Status: 16640/50000 (33%) | Loss: 1.613488
Train Epoch: 6 | Batch Status: 17280/50000 (35%) | Loss: 1.542426
Train Epoch: 6 | Batch Status: 17920/50000 (36%) | Loss: 1.535990
Train Epoch: 6 | Batch Status: 18560/50000 (37%) | Loss: 1.494462
Train Epoch: 6 | Batch Status: 19200/50000 (38%) | Loss: 1.590418
Train Epoch: 6 | Batch Status: 19840/50000 (40%) | Loss: 1.251730
Train Epoch: 6 | Batch Status: 20480/50000 (41%) | Loss: 1.378694
Train Epoch: 6 | Batch Status: 21120/50000 (42%) | Loss: 1.516298
Train Epoch: 6 | Batch Status: 21760/50000 (43%) | Loss: 1.458833
Train Epoch: 6 | Batch Status: 22400/50000 (45%) | Loss: 1.359884
Train Epoch: 6 | Batch Status: 23040/50000 (46%) | Loss: 1.403825
Train Epoch: 6 | Batch Status: 23680/50000 (47%) | Loss: 1.728887
Train Epoch: 6 | Batch Status: 24320/50000 (49%) | Loss: 1.594227
Train Epoch: 6 | Batch Status: 24960/50000 (50%) | Loss: 1.758768
Train Epoch: 6 | Batch Status: 25600/50000 (51%) | Loss: 1.736555
Train Epoch: 6 | Batch Status: 26240/50000 (52%) | Loss: 1.703857
Train Epoch: 6 | Batch Status: 26880/50000 (54%) | Loss: 1.770282
Train Epoch: 6 | Batch Status: 27520/50000 (55%) | Loss: 1.529734
Train Epoch: 6 | Batch Status: 28160/50000 (56%) | Loss: 1.475853
Train Epoch: 6 | Batch Status: 28800/50000 (58%) | Loss: 1.521284
Train Epoch: 6 | Batch Status: 29440/50000 (59%) | Loss: 1.516390
Train Epoch: 6 | Batch Status: 30080/50000 (60%) | Loss: 1.471442
Train Epoch: 6 | Batch Status: 30720/50000 (61%) | Loss: 1.447698
Train Epoch: 6 | Batch Status: 31360/50000 (63%) | Loss: 1.463640
Train Epoch: 6 | Batch Status: 32000/50000 (64%) | Loss: 1.282755
Train Epoch: 6 | Batch Status: 32640/50000 (65%) | Loss: 1.409514
Train Epoch: 6 | Batch Status: 33280/50000 (66%) | Loss: 1.555134
Train Epoch: 6 | Batch Status: 33920/50000 (68%) | Loss: 1.548699
Train Epoch: 6 | Batch Status: 34560/50000 (69%) | Loss: 1.323416
Train Epoch: 6 | Batch Status: 35200/50000 (70%) | Loss: 1.287253
Train Epoch: 6 | Batch Status: 35840/50000 (72%) | Loss: 1.565480
Train Epoch: 6 | Batch Status: 36480/50000 (73%) | Loss: 1.600028
Train Epoch: 6 | Batch Status: 37120/50000 (74%) | Loss: 1.258337
Train Epoch: 6 | Batch Status: 37760/50000 (75%) | Loss: 1.535311
Train Epoch: 6 | Batch Status: 38400/50000 (77%) | Loss: 1.336536
Train Epoch: 6 | Batch Status: 39040/50000 (78%) | Loss: 1.549631
Train Epoch: 6 | Batch Status: 39680/50000 (79%) | Loss: 1.815929
Train Epoch: 6 | Batch Status: 40320/50000 (81%) | Loss: 1.454838
Train Epoch: 6 | Batch Status: 40960/50000 (82%) | Loss: 1.409704
Train Epoch: 6 | Batch Status: 41600/50000 (83%) | Loss: 1.490758
Train Epoch: 6 | Batch Status: 42240/50000 (84%) | Loss: 1.388862
Train Epoch: 6 | Batch Status: 42880/50000 (86%) | Loss: 1.550239
Train Epoch: 6 | Batch Status: 43520/50000 (87%) | Loss: 1.652522
Train Epoch: 6 | Batch Status: 44160/50000 (88%) | Loss: 1.589689
Train Epoch: 6 | Batch Status: 44800/50000 (90%) | Loss: 1.316224
Train Epoch: 6 | Batch Status: 45440/50000 (91%) | Loss: 1.647762
Train Epoch: 6 | Batch Status: 46080/50000 (92%) | Loss: 1.556998
Train Epoch: 6 | Batch Status: 46720/50000 (93%) | Loss: 1.372136
Train Epoch: 6 | Batch Status: 47360/50000 (95%) | Loss: 1.427144
Train Epoch: 6 | Batch Status: 48000/50000 (96%) | Loss: 1.546011
Train Epoch: 6 | Batch Status: 48640/50000 (97%) | Loss: 1.406316
Train Epoch: 6 | Batch Status: 49280/50000 (98%) | Loss: 1.570122
Train Epoch: 6 | Batch Status: 49920/50000 (100%) | Loss: 1.342062
Training time: 0m 23s
===========================
Test set: Average loss: 0.0235, Accuracy: 4566/10000 (46%)
Testing time: 0m 25s
Train Epoch: 7 | Batch Status: 0/50000 (0%) | Loss: 1.562239
Train Epoch: 7 | Batch Status: 640/50000 (1%) | Loss: 1.569955
Train Epoch: 7 | Batch Status: 1280/50000 (3%) | Loss: 1.373185
Train Epoch: 7 | Batch Status: 1920/50000 (4%) | Loss: 1.410155
Train Epoch: 7 | Batch Status: 2560/50000 (5%) | Loss: 1.332710
Train Epoch: 7 | Batch Status: 3200/50000 (6%) | Loss: 1.641574
Train Epoch: 7 | Batch Status: 3840/50000 (8%) | Loss: 1.382403
Train Epoch: 7 | Batch Status: 4480/50000 (9%) | Loss: 1.438546
Train Epoch: 7 | Batch Status: 5120/50000 (10%) | Loss: 1.288627
Train Epoch: 7 | Batch Status: 5760/50000 (12%) | Loss: 1.271840
Train Epoch: 7 | Batch Status: 6400/50000 (13%) | Loss: 1.608251
Train Epoch: 7 | Batch Status: 7040/50000 (14%) | Loss: 1.499631
Train Epoch: 7 | Batch Status: 7680/50000 (15%) | Loss: 1.474010
Train Epoch: 7 | Batch Status: 8320/50000 (17%) | Loss: 1.499393
Train Epoch: 7 | Batch Status: 8960/50000 (18%) | Loss: 1.656967
Train Epoch: 7 | Batch Status: 9600/50000 (19%) | Loss: 1.393264
Train Epoch: 7 | Batch Status: 10240/50000 (20%) | Loss: 1.507577
Train Epoch: 7 | Batch Status: 10880/50000 (22%) | Loss: 1.550803
Train Epoch: 7 | Batch Status: 11520/50000 (23%) | Loss: 1.349329
Train Epoch: 7 | Batch Status: 12160/50000 (24%) | Loss: 1.265439
Train Epoch: 7 | Batch Status: 12800/50000 (26%) | Loss: 1.291240
Train Epoch: 7 | Batch Status: 13440/50000 (27%) | Loss: 1.801088
Train Epoch: 7 | Batch Status: 14080/50000 (28%) | Loss: 1.634308
Train Epoch: 7 | Batch Status: 14720/50000 (29%) | Loss: 1.692548
Train Epoch: 7 | Batch Status: 15360/50000 (31%) | Loss: 1.441648
Train Epoch: 7 | Batch Status: 16000/50000 (32%) | Loss: 1.304116
Train Epoch: 7 | Batch Status: 16640/50000 (33%) | Loss: 1.506575
Train Epoch: 7 | Batch Status: 17280/50000 (35%) | Loss: 1.246030
Train Epoch: 7 | Batch Status: 17920/50000 (36%) | Loss: 1.335918
Train Epoch: 7 | Batch Status: 18560/50000 (37%) | Loss: 1.490748
Train Epoch: 7 | Batch Status: 19200/50000 (38%) | Loss: 1.352653
Train Epoch: 7 | Batch Status: 19840/50000 (40%) | Loss: 1.585139
Train Epoch: 7 | Batch Status: 20480/50000 (41%) | Loss: 1.420626
Train Epoch: 7 | Batch Status: 21120/50000 (42%) | Loss: 1.134602
Train Epoch: 7 | Batch Status: 21760/50000 (43%) | Loss: 1.391747
Train Epoch: 7 | Batch Status: 22400/50000 (45%) | Loss: 1.402865
Train Epoch: 7 | Batch Status: 23040/50000 (46%) | Loss: 1.225908
Train Epoch: 7 | Batch Status: 23680/50000 (47%) | Loss: 1.368797
Train Epoch: 7 | Batch Status: 24320/50000 (49%) | Loss: 1.585011
Train Epoch: 7 | Batch Status: 24960/50000 (50%) | Loss: 1.497884
Train Epoch: 7 | Batch Status: 25600/50000 (51%) | Loss: 1.527951
Train Epoch: 7 | Batch Status: 26240/50000 (52%) | Loss: 1.400615
Train Epoch: 7 | Batch Status: 26880/50000 (54%) | Loss: 1.637190
Train Epoch: 7 | Batch Status: 27520/50000 (55%) | Loss: 1.483005
Train Epoch: 7 | Batch Status: 28160/50000 (56%) | Loss: 1.378627
Train Epoch: 7 | Batch Status: 28800/50000 (58%) | Loss: 1.362520
Train Epoch: 7 | Batch Status: 29440/50000 (59%) | Loss: 1.452656
Train Epoch: 7 | Batch Status: 30080/50000 (60%) | Loss: 1.494902
Train Epoch: 7 | Batch Status: 30720/50000 (61%) | Loss: 1.262069
Train Epoch: 7 | Batch Status: 31360/50000 (63%) | Loss: 1.357809
Train Epoch: 7 | Batch Status: 32000/50000 (64%) | Loss: 1.439007
Train Epoch: 7 | Batch Status: 32640/50000 (65%) | Loss: 1.338017
Train Epoch: 7 | Batch Status: 33280/50000 (66%) | Loss: 1.234503
Train Epoch: 7 | Batch Status: 33920/50000 (68%) | Loss: 1.467465
Train Epoch: 7 | Batch Status: 34560/50000 (69%) | Loss: 1.522013
Train Epoch: 7 | Batch Status: 35200/50000 (70%) | Loss: 1.517237
Train Epoch: 7 | Batch Status: 35840/50000 (72%) | Loss: 1.385965
Train Epoch: 7 | Batch Status: 36480/50000 (73%) | Loss: 1.324558
Train Epoch: 7 | Batch Status: 37120/50000 (74%) | Loss: 1.300768
Train Epoch: 7 | Batch Status: 37760/50000 (75%) | Loss: 1.271977
Train Epoch: 7 | Batch Status: 38400/50000 (77%) | Loss: 1.425601
Train Epoch: 7 | Batch Status: 39040/50000 (78%) | Loss: 1.487182
Train Epoch: 7 | Batch Status: 39680/50000 (79%) | Loss: 1.477497
Train Epoch: 7 | Batch Status: 40320/50000 (81%) | Loss: 1.484149
Train Epoch: 7 | Batch Status: 40960/50000 (82%) | Loss: 1.628929
Train Epoch: 7 | Batch Status: 41600/50000 (83%) | Loss: 1.313202
Train Epoch: 7 | Batch Status: 42240/50000 (84%) | Loss: 1.453108
Train Epoch: 7 | Batch Status: 42880/50000 (86%) | Loss: 1.417666
Train Epoch: 7 | Batch Status: 43520/50000 (87%) | Loss: 1.311010
Train Epoch: 7 | Batch Status: 44160/50000 (88%) | Loss: 1.229746
Train Epoch: 7 | Batch Status: 44800/50000 (90%) | Loss: 1.265645
Train Epoch: 7 | Batch Status: 45440/50000 (91%) | Loss: 1.431317
Train Epoch: 7 | Batch Status: 46080/50000 (92%) | Loss: 1.209915
Train Epoch: 7 | Batch Status: 46720/50000 (93%) | Loss: 1.325422
Train Epoch: 7 | Batch Status: 47360/50000 (95%) | Loss: 1.287719
Train Epoch: 7 | Batch Status: 48000/50000 (96%) | Loss: 1.390214
Train Epoch: 7 | Batch Status: 48640/50000 (97%) | Loss: 1.613132
Train Epoch: 7 | Batch Status: 49280/50000 (98%) | Loss: 1.481730
Train Epoch: 7 | Batch Status: 49920/50000 (100%) | Loss: 1.740285
Training time: 0m 21s
===========================
Test set: Average loss: 0.0230, Accuracy: 4707/10000 (47%)
Testing time: 0m 24s
Train Epoch: 8 | Batch Status: 0/50000 (0%) | Loss: 1.490485
Train Epoch: 8 | Batch Status: 640/50000 (1%) | Loss: 1.405304
Train Epoch: 8 | Batch Status: 1280/50000 (3%) | Loss: 1.562250
Train Epoch: 8 | Batch Status: 1920/50000 (4%) | Loss: 1.306864
Train Epoch: 8 | Batch Status: 2560/50000 (5%) | Loss: 1.432374
Train Epoch: 8 | Batch Status: 3200/50000 (6%) | Loss: 1.255420
Train Epoch: 8 | Batch Status: 3840/50000 (8%) | Loss: 1.350601
Train Epoch: 8 | Batch Status: 4480/50000 (9%) | Loss: 1.467811
Train Epoch: 8 | Batch Status: 5120/50000 (10%) | Loss: 1.248755
Train Epoch: 8 | Batch Status: 5760/50000 (12%) | Loss: 1.438931
Train Epoch: 8 | Batch Status: 6400/50000 (13%) | Loss: 1.450226
Train Epoch: 8 | Batch Status: 7040/50000 (14%) | Loss: 1.456915
Train Epoch: 8 | Batch Status: 7680/50000 (15%) | Loss: 1.384522
Train Epoch: 8 | Batch Status: 8320/50000 (17%) | Loss: 1.439070
Train Epoch: 8 | Batch Status: 8960/50000 (18%) | Loss: 1.264879
Train Epoch: 8 | Batch Status: 9600/50000 (19%) | Loss: 1.207113
Train Epoch: 8 | Batch Status: 10240/50000 (20%) | Loss: 1.337852
Train Epoch: 8 | Batch Status: 10880/50000 (22%) | Loss: 1.416937
Train Epoch: 8 | Batch Status: 11520/50000 (23%) | Loss: 1.330931
Train Epoch: 8 | Batch Status: 12160/50000 (24%) | Loss: 1.735223
Train Epoch: 8 | Batch Status: 12800/50000 (26%) | Loss: 1.346807
Train Epoch: 8 | Batch Status: 13440/50000 (27%) | Loss: 1.362811
Train Epoch: 8 | Batch Status: 14080/50000 (28%) | Loss: 1.272877
Train Epoch: 8 | Batch Status: 14720/50000 (29%) | Loss: 1.458186
Train Epoch: 8 | Batch Status: 15360/50000 (31%) | Loss: 1.572726
Train Epoch: 8 | Batch Status: 16000/50000 (32%) | Loss: 1.397675
Train Epoch: 8 | Batch Status: 16640/50000 (33%) | Loss: 1.515537
Train Epoch: 8 | Batch Status: 17280/50000 (35%) | Loss: 1.442986
Train Epoch: 8 | Batch Status: 17920/50000 (36%) | Loss: 1.665062
Train Epoch: 8 | Batch Status: 18560/50000 (37%) | Loss: 1.430162
Train Epoch: 8 | Batch Status: 19200/50000 (38%) | Loss: 1.449473
Train Epoch: 8 | Batch Status: 19840/50000 (40%) | Loss: 1.339123
Train Epoch: 8 | Batch Status: 20480/50000 (41%) | Loss: 1.208991
Train Epoch: 8 | Batch Status: 21120/50000 (42%) | Loss: 1.313727
Train Epoch: 8 | Batch Status: 21760/50000 (43%) | Loss: 1.374885
Train Epoch: 8 | Batch Status: 22400/50000 (45%) | Loss: 1.199580
Train Epoch: 8 | Batch Status: 23040/50000 (46%) | Loss: 1.363010
Train Epoch: 8 | Batch Status: 23680/50000 (47%) | Loss: 1.338400
Train Epoch: 8 | Batch Status: 24320/50000 (49%) | Loss: 1.852048
Train Epoch: 8 | Batch Status: 24960/50000 (50%) | Loss: 1.508676
Train Epoch: 8 | Batch Status: 25600/50000 (51%) | Loss: 1.341982
Train Epoch: 8 | Batch Status: 26240/50000 (52%) | Loss: 1.491034
Train Epoch: 8 | Batch Status: 26880/50000 (54%) | Loss: 1.453443
Train Epoch: 8 | Batch Status: 27520/50000 (55%) | Loss: 1.535265
Train Epoch: 8 | Batch Status: 28160/50000 (56%) | Loss: 1.701670
Train Epoch: 8 | Batch Status: 28800/50000 (58%) | Loss: 1.470390
Train Epoch: 8 | Batch Status: 29440/50000 (59%) | Loss: 1.214476
Train Epoch: 8 | Batch Status: 30080/50000 (60%) | Loss: 1.393916
Train Epoch: 8 | Batch Status: 30720/50000 (61%) | Loss: 1.430149
Train Epoch: 8 | Batch Status: 31360/50000 (63%) | Loss: 1.341408
Train Epoch: 8 | Batch Status: 32000/50000 (64%) | Loss: 1.482774
Train Epoch: 8 | Batch Status: 32640/50000 (65%) | Loss: 1.293424
Train Epoch: 8 | Batch Status: 33280/50000 (66%) | Loss: 1.324275
Train Epoch: 8 | Batch Status: 33920/50000 (68%) | Loss: 1.290158
Train Epoch: 8 | Batch Status: 34560/50000 (69%) | Loss: 1.570468
Train Epoch: 8 | Batch Status: 35200/50000 (70%) | Loss: 1.427853
Train Epoch: 8 | Batch Status: 35840/50000 (72%) | Loss: 1.446951
Train Epoch: 8 | Batch Status: 36480/50000 (73%) | Loss: 1.318074
Train Epoch: 8 | Batch Status: 37120/50000 (74%) | Loss: 1.289461
Train Epoch: 8 | Batch Status: 37760/50000 (75%) | Loss: 1.499506
Train Epoch: 8 | Batch Status: 38400/50000 (77%) | Loss: 1.604873
Train Epoch: 8 | Batch Status: 39040/50000 (78%) | Loss: 1.452628
Train Epoch: 8 | Batch Status: 39680/50000 (79%) | Loss: 1.308516
Train Epoch: 8 | Batch Status: 40320/50000 (81%) | Loss: 1.285841
Train Epoch: 8 | Batch Status: 40960/50000 (82%) | Loss: 1.457001
Train Epoch: 8 | Batch Status: 41600/50000 (83%) | Loss: 1.387827
Train Epoch: 8 | Batch Status: 42240/50000 (84%) | Loss: 1.327831
Train Epoch: 8 | Batch Status: 42880/50000 (86%) | Loss: 1.421791
Train Epoch: 8 | Batch Status: 43520/50000 (87%) | Loss: 1.443411
Train Epoch: 8 | Batch Status: 44160/50000 (88%) | Loss: 1.384994
Train Epoch: 8 | Batch Status: 44800/50000 (90%) | Loss: 1.261866
Train Epoch: 8 | Batch Status: 45440/50000 (91%) | Loss: 1.374584
Train Epoch: 8 | Batch Status: 46080/50000 (92%) | Loss: 1.558956
Train Epoch: 8 | Batch Status: 46720/50000 (93%) | Loss: 1.506768
Train Epoch: 8 | Batch Status: 47360/50000 (95%) | Loss: 1.582520
Train Epoch: 8 | Batch Status: 48000/50000 (96%) | Loss: 1.457602
Train Epoch: 8 | Batch Status: 48640/50000 (97%) | Loss: 1.142487
Train Epoch: 8 | Batch Status: 49280/50000 (98%) | Loss: 1.414992
Train Epoch: 8 | Batch Status: 49920/50000 (100%) | Loss: 1.621371
Training time: 0m 26s
===========================
Test set: Average loss: 0.0247, Accuracy: 4356/10000 (44%)
Testing time: 0m 28s
Train Epoch: 9 | Batch Status: 0/50000 (0%) | Loss: 1.549901
Train Epoch: 9 | Batch Status: 640/50000 (1%) | Loss: 1.228625
Train Epoch: 9 | Batch Status: 1280/50000 (3%) | Loss: 1.321341
Train Epoch: 9 | Batch Status: 1920/50000 (4%) | Loss: 1.563262
Train Epoch: 9 | Batch Status: 2560/50000 (5%) | Loss: 1.349002
Train Epoch: 9 | Batch Status: 3200/50000 (6%) | Loss: 1.468838
Train Epoch: 9 | Batch Status: 3840/50000 (8%) | Loss: 1.504409
Train Epoch: 9 | Batch Status: 4480/50000 (9%) | Loss: 1.399260
Train Epoch: 9 | Batch Status: 5120/50000 (10%) | Loss: 1.374454
Train Epoch: 9 | Batch Status: 5760/50000 (12%) | Loss: 1.412114
Train Epoch: 9 | Batch Status: 6400/50000 (13%) | Loss: 1.341410
Train Epoch: 9 | Batch Status: 7040/50000 (14%) | Loss: 1.536582
Train Epoch: 9 | Batch Status: 7680/50000 (15%) | Loss: 1.478322
Train Epoch: 9 | Batch Status: 8320/50000 (17%) | Loss: 1.390092
Train Epoch: 9 | Batch Status: 8960/50000 (18%) | Loss: 1.615571
Train Epoch: 9 | Batch Status: 9600/50000 (19%) | Loss: 1.382569
Train Epoch: 9 | Batch Status: 10240/50000 (20%) | Loss: 1.250875
Train Epoch: 9 | Batch Status: 10880/50000 (22%) | Loss: 1.168064
Train Epoch: 9 | Batch Status: 11520/50000 (23%) | Loss: 1.339164
Train Epoch: 9 | Batch Status: 12160/50000 (24%) | Loss: 1.434756
Train Epoch: 9 | Batch Status: 12800/50000 (26%) | Loss: 1.218741
Train Epoch: 9 | Batch Status: 13440/50000 (27%) | Loss: 1.319469
Train Epoch: 9 | Batch Status: 14080/50000 (28%) | Loss: 1.318788
Train Epoch: 9 | Batch Status: 14720/50000 (29%) | Loss: 1.309674
Train Epoch: 9 | Batch Status: 15360/50000 (31%) | Loss: 1.436764
Train Epoch: 9 | Batch Status: 16000/50000 (32%) | Loss: 1.096590
Train Epoch: 9 | Batch Status: 16640/50000 (33%) | Loss: 1.358871
Train Epoch: 9 | Batch Status: 17280/50000 (35%) | Loss: 1.150054
Train Epoch: 9 | Batch Status: 17920/50000 (36%) | Loss: 1.411152
Train Epoch: 9 | Batch Status: 18560/50000 (37%) | Loss: 1.245735
Train Epoch: 9 | Batch Status: 19200/50000 (38%) | Loss: 1.291777
Train Epoch: 9 | Batch Status: 19840/50000 (40%) | Loss: 1.146520
Train Epoch: 9 | Batch Status: 20480/50000 (41%) | Loss: 1.399461
Train Epoch: 9 | Batch Status: 21120/50000 (42%) | Loss: 1.380928
Train Epoch: 9 | Batch Status: 21760/50000 (43%) | Loss: 1.513975
Train Epoch: 9 | Batch Status: 22400/50000 (45%) | Loss: 1.253331
Train Epoch: 9 | Batch Status: 23040/50000 (46%) | Loss: 1.608873
Train Epoch: 9 | Batch Status: 23680/50000 (47%) | Loss: 1.337889
Train Epoch: 9 | Batch Status: 24320/50000 (49%) | Loss: 1.368205
Train Epoch: 9 | Batch Status: 24960/50000 (50%) | Loss: 1.327058
Train Epoch: 9 | Batch Status: 25600/50000 (51%) | Loss: 1.388922
Train Epoch: 9 | Batch Status: 26240/50000 (52%) | Loss: 1.399372
Train Epoch: 9 | Batch Status: 26880/50000 (54%) | Loss: 1.416162
Train Epoch: 9 | Batch Status: 27520/50000 (55%) | Loss: 1.741715
Train Epoch: 9 | Batch Status: 28160/50000 (56%) | Loss: 1.732748
Train Epoch: 9 | Batch Status: 28800/50000 (58%) | Loss: 1.312102
Train Epoch: 9 | Batch Status: 29440/50000 (59%) | Loss: 1.568573
Train Epoch: 9 | Batch Status: 30080/50000 (60%) | Loss: 1.343268
Train Epoch: 9 | Batch Status: 30720/50000 (61%) | Loss: 1.371774
Train Epoch: 9 | Batch Status: 31360/50000 (63%) | Loss: 1.386914
Train Epoch: 9 | Batch Status: 32000/50000 (64%) | Loss: 1.439681
Train Epoch: 9 | Batch Status: 32640/50000 (65%) | Loss: 1.326122
Train Epoch: 9 | Batch Status: 33280/50000 (66%) | Loss: 1.479827
Train Epoch: 9 | Batch Status: 33920/50000 (68%) | Loss: 1.414823
Train Epoch: 9 | Batch Status: 34560/50000 (69%) | Loss: 1.553213
Train Epoch: 9 | Batch Status: 35200/50000 (70%) | Loss: 1.633047
Train Epoch: 9 | Batch Status: 35840/50000 (72%) | Loss: 1.228807
Train Epoch: 9 | Batch Status: 36480/50000 (73%) | Loss: 1.438715
Train Epoch: 9 | Batch Status: 37120/50000 (74%) | Loss: 1.124521
Train Epoch: 9 | Batch Status: 37760/50000 (75%) | Loss: 1.458118
Train Epoch: 9 | Batch Status: 38400/50000 (77%) | Loss: 1.238657
Train Epoch: 9 | Batch Status: 39040/50000 (78%) | Loss: 1.509338
Train Epoch: 9 | Batch Status: 39680/50000 (79%) | Loss: 1.263390
Train Epoch: 9 | Batch Status: 40320/50000 (81%) | Loss: 1.517283
Train Epoch: 9 | Batch Status: 40960/50000 (82%) | Loss: 1.423136
Train Epoch: 9 | Batch Status: 41600/50000 (83%) | Loss: 1.319422
Train Epoch: 9 | Batch Status: 42240/50000 (84%) | Loss: 1.094402
Train Epoch: 9 | Batch Status: 42880/50000 (86%) | Loss: 1.400520
Train Epoch: 9 | Batch Status: 43520/50000 (87%) | Loss: 1.400982
Train Epoch: 9 | Batch Status: 44160/50000 (88%) | Loss: 1.480996
Train Epoch: 9 | Batch Status: 44800/50000 (90%) | Loss: 1.164408
Train Epoch: 9 | Batch Status: 45440/50000 (91%) | Loss: 1.398701
Train Epoch: 9 | Batch Status: 46080/50000 (92%) | Loss: 1.458858
Train Epoch: 9 | Batch Status: 46720/50000 (93%) | Loss: 1.547484
Train Epoch: 9 | Batch Status: 47360/50000 (95%) | Loss: 1.586004
Train Epoch: 9 | Batch Status: 48000/50000 (96%) | Loss: 1.704248
Train Epoch: 9 | Batch Status: 48640/50000 (97%) | Loss: 1.362222
Train Epoch: 9 | Batch Status: 49280/50000 (98%) | Loss: 1.287277
Train Epoch: 9 | Batch Status: 49920/50000 (100%) | Loss: 1.374157
Training time: 0m 22s
===========================
Test set: Average loss: 0.0224, Accuracy: 5003/10000 (50%)
Testing time: 0m 24s
Total Time: 3m 44s
Model was trained on cpu!

Process finished with exit code 0

      
