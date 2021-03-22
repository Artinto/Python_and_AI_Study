from __future__ import print_function   
from torch import nn, optim, cuda             # cuda : GPU 사용 
from torch.utils import data
from torchvision import datasets, transforms
import torch.nn.functional as F
import time
import tensorflow as tf





batch_size = 64                                               #batch size 64로
device = 'cuda' if cuda.is_available() else 'cpu'             #cuda 사용가능시 gpu 불가능시 cpu
print(f'Training CIFAR10 Model on {device}\n{"=" * 44}')


train_dataset = datasets.CIFAR10(root='./data',train=True,transform=transforms.ToTensor(),download=True) 

  #  dataset의 root directory,  Tensor type 으로 변환 , True -> 인터넷으로부터 dataset 다운로드 

test_dataset = datasets.CIFAR10(root='./data',train=False,transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(32*32*3, 2024)  # 7개 층으로 이루어진 모델 생성 (input 3*32*32 , output10)
        self.l2 = nn.Linear(2024, 1512)
        self.l3 = nn.Linear(1512, 1000)
        self.l4 = nn.Linear(1000, 512)
        self.l5 = nn.Linear(512, 256)
        self.l6 = nn.Linear(256, 128)
        self.l7 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 32*32*3)  # Flatten the data (n, 3, 32, 32)-> (n, 32*32*3)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = F.relu(self.l5(x))
        x = F.relu(self.l6(x))
        return self.l7(x)


model = Net()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.7)
#m 0.01/0.5 40%   0.01/0.7 48%    0.05/07 44  0.05/0.5 48

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
'''Training CIFAR10 Model on cuda
============================================
Files already downloaded and verified
Dataset CIFAR10
    Number of datapoints: 50000
    Root location: ./data
    Split: Train
    StandardTransform
Transform: ToTensor()
Train Epoch: 1 | Batch Status: 0/50000 (0%) | Loss: 2.299610
Train Epoch: 1 | Batch Status: 640/50000 (1%) | Loss: 2.305643
Train Epoch: 1 | Batch Status: 1280/50000 (3%) | Loss: 2.313338
Train Epoch: 1 | Batch Status: 1920/50000 (4%) | Loss: 2.303032
Train Epoch: 1 | Batch Status: 2560/50000 (5%) | Loss: 2.295318
Train Epoch: 1 | Batch Status: 3200/50000 (6%) | Loss: 2.310917
Train Epoch: 1 | Batch Status: 3840/50000 (8%) | Loss: 2.307976
Train Epoch: 1 | Batch Status: 4480/50000 (9%) | Loss: 2.295258
Train Epoch: 1 | Batch Status: 5120/50000 (10%) | Loss: 2.304300
Train Epoch: 1 | Batch Status: 5760/50000 (12%) | Loss: 2.297152
Train Epoch: 1 | Batch Status: 6400/50000 (13%) | Loss: 2.304467
Train Epoch: 1 | Batch Status: 7040/50000 (14%) | Loss: 2.291240
Train Epoch: 1 | Batch Status: 7680/50000 (15%) | Loss: 2.302263
Train Epoch: 1 | Batch Status: 8320/50000 (17%) | Loss: 2.297885
Train Epoch: 1 | Batch Status: 8960/50000 (18%) | Loss: 2.295233
Train Epoch: 1 | Batch Status: 9600/50000 (19%) | Loss: 2.296494
Train Epoch: 1 | Batch Status: 10240/50000 (20%) | Loss: 2.306910
Train Epoch: 1 | Batch Status: 10880/50000 (22%) | Loss: 2.289648
Train Epoch: 1 | Batch Status: 11520/50000 (23%) | Loss: 2.304942
Train Epoch: 1 | Batch Status: 12160/50000 (24%) | Loss: 2.301336
Train Epoch: 1 | Batch Status: 12800/50000 (26%) | Loss: 2.303797
Train Epoch: 1 | Batch Status: 13440/50000 (27%) | Loss: 2.295525
Train Epoch: 1 | Batch Status: 14080/50000 (28%) | Loss: 2.300548
Train Epoch: 1 | Batch Status: 14720/50000 (29%) | Loss: 2.285904
Train Epoch: 1 | Batch Status: 15360/50000 (31%) | Loss: 2.295567
Train Epoch: 1 | Batch Status: 16000/50000 (32%) | Loss: 2.299388
Train Epoch: 1 | Batch Status: 16640/50000 (33%) | Loss: 2.293376
Train Epoch: 1 | Batch Status: 17280/50000 (35%) | Loss: 2.286936
Train Epoch: 1 | Batch Status: 17920/50000 (36%) | Loss: 2.269431
Train Epoch: 1 | Batch Status: 18560/50000 (37%) | Loss: 2.273850
Train Epoch: 1 | Batch Status: 19200/50000 (38%) | Loss: 2.218840
Train Epoch: 1 | Batch Status: 19840/50000 (40%) | Loss: 2.181503
Train Epoch: 1 | Batch Status: 20480/50000 (41%) | Loss: 2.171349
Train Epoch: 1 | Batch Status: 21120/50000 (42%) | Loss: 2.105021
Train Epoch: 1 | Batch Status: 21760/50000 (43%) | Loss: 2.069272
Train Epoch: 1 | Batch Status: 22400/50000 (45%) | Loss: 2.037454
Train Epoch: 1 | Batch Status: 23040/50000 (46%) | Loss: 2.169995
Train Epoch: 1 | Batch Status: 23680/50000 (47%) | Loss: 2.100239
Train Epoch: 1 | Batch Status: 24320/50000 (49%) | Loss: 2.242068
Train Epoch: 1 | Batch Status: 24960/50000 (50%) | Loss: 2.074023
Train Epoch: 1 | Batch Status: 25600/50000 (51%) | Loss: 2.108801
Train Epoch: 1 | Batch Status: 26240/50000 (52%) | Loss: 2.090972
Train Epoch: 1 | Batch Status: 26880/50000 (54%) | Loss: 2.005067
Train Epoch: 1 | Batch Status: 27520/50000 (55%) | Loss: 2.069057
Train Epoch: 1 | Batch Status: 28160/50000 (56%) | Loss: 2.021923
Train Epoch: 1 | Batch Status: 28800/50000 (58%) | Loss: 2.017974
Train Epoch: 1 | Batch Status: 29440/50000 (59%) | Loss: 2.073184
Train Epoch: 1 | Batch Status: 30080/50000 (60%) | Loss: 2.150319
Train Epoch: 1 | Batch Status: 30720/50000 (61%) | Loss: 2.114726
Train Epoch: 1 | Batch Status: 31360/50000 (63%) | Loss: 2.012850
Train Epoch: 1 | Batch Status: 32000/50000 (64%) | Loss: 2.017096
Train Epoch: 1 | Batch Status: 32640/50000 (65%) | Loss: 1.925617
Train Epoch: 1 | Batch Status: 33280/50000 (66%) | Loss: 2.094115
Train Epoch: 1 | Batch Status: 33920/50000 (68%) | Loss: 1.833734
Train Epoch: 1 | Batch Status: 34560/50000 (69%) | Loss: 1.883199
Train Epoch: 1 | Batch Status: 35200/50000 (70%) | Loss: 1.969790
Train Epoch: 1 | Batch Status: 35840/50000 (72%) | Loss: 2.040019
Train Epoch: 1 | Batch Status: 36480/50000 (73%) | Loss: 1.951953
Train Epoch: 1 | Batch Status: 37120/50000 (74%) | Loss: 2.099515
Train Epoch: 1 | Batch Status: 37760/50000 (75%) | Loss: 1.868864
Train Epoch: 1 | Batch Status: 38400/50000 (77%) | Loss: 2.048178
Train Epoch: 1 | Batch Status: 39040/50000 (78%) | Loss: 2.034472
Train Epoch: 1 | Batch Status: 39680/50000 (79%) | Loss: 1.977495
Train Epoch: 1 | Batch Status: 40320/50000 (81%) | Loss: 2.000459
Train Epoch: 1 | Batch Status: 40960/50000 (82%) | Loss: 2.147420
Train Epoch: 1 | Batch Status: 41600/50000 (83%) | Loss: 2.058957
Train Epoch: 1 | Batch Status: 42240/50000 (84%) | Loss: 1.876781
Train Epoch: 1 | Batch Status: 42880/50000 (86%) | Loss: 2.014632
Train Epoch: 1 | Batch Status: 43520/50000 (87%) | Loss: 2.038451
Train Epoch: 1 | Batch Status: 44160/50000 (88%) | Loss: 1.992280
Train Epoch: 1 | Batch Status: 44800/50000 (90%) | Loss: 2.056334
Train Epoch: 1 | Batch Status: 45440/50000 (91%) | Loss: 1.886224
Train Epoch: 1 | Batch Status: 46080/50000 (92%) | Loss: 2.023825
Train Epoch: 1 | Batch Status: 46720/50000 (93%) | Loss: 1.795415
Train Epoch: 1 | Batch Status: 47360/50000 (95%) | Loss: 1.983236
Train Epoch: 1 | Batch Status: 48000/50000 (96%) | Loss: 1.958078
Train Epoch: 1 | Batch Status: 48640/50000 (97%) | Loss: 1.802577
Train Epoch: 1 | Batch Status: 49280/50000 (98%) | Loss: 1.881634
Train Epoch: 1 | Batch Status: 49920/50000 (100%) | Loss: 1.934159
Training time: 0m 10s
===========================
Test set: Average loss: 0.0302, Accuracy: 2568/10000 (26%)
Testing time: 0m 12s
Train Epoch: 2 | Batch Status: 0/50000 (0%) | Loss: 1.965283
Train Epoch: 2 | Batch Status: 640/50000 (1%) | Loss: 1.850700
Train Epoch: 2 | Batch Status: 1280/50000 (3%) | Loss: 1.930664
Train Epoch: 2 | Batch Status: 1920/50000 (4%) | Loss: 1.910811
Train Epoch: 2 | Batch Status: 2560/50000 (5%) | Loss: 1.823683
Train Epoch: 2 | Batch Status: 3200/50000 (6%) | Loss: 1.931850
Train Epoch: 2 | Batch Status: 3840/50000 (8%) | Loss: 1.897761
Train Epoch: 2 | Batch Status: 4480/50000 (9%) | Loss: 1.679176
Train Epoch: 2 | Batch Status: 5120/50000 (10%) | Loss: 1.852659
Train Epoch: 2 | Batch Status: 5760/50000 (12%) | Loss: 2.027464
Train Epoch: 2 | Batch Status: 6400/50000 (13%) | Loss: 1.769996
Train Epoch: 2 | Batch Status: 7040/50000 (14%) | Loss: 1.941267
Train Epoch: 2 | Batch Status: 7680/50000 (15%) | Loss: 1.934577
Train Epoch: 2 | Batch Status: 8320/50000 (17%) | Loss: 2.011495
Train Epoch: 2 | Batch Status: 8960/50000 (18%) | Loss: 1.896703
Train Epoch: 2 | Batch Status: 9600/50000 (19%) | Loss: 1.830372
Train Epoch: 2 | Batch Status: 10240/50000 (20%) | Loss: 1.956248
Train Epoch: 2 | Batch Status: 10880/50000 (22%) | Loss: 1.946238
Train Epoch: 2 | Batch Status: 11520/50000 (23%) | Loss: 1.853006
Train Epoch: 2 | Batch Status: 12160/50000 (24%) | Loss: 2.076060
Train Epoch: 2 | Batch Status: 12800/50000 (26%) | Loss: 1.841104
Train Epoch: 2 | Batch Status: 13440/50000 (27%) | Loss: 1.834171
Train Epoch: 2 | Batch Status: 14080/50000 (28%) | Loss: 1.865260
Train Epoch: 2 | Batch Status: 14720/50000 (29%) | Loss: 1.979973
Train Epoch: 2 | Batch Status: 15360/50000 (31%) | Loss: 1.882024
Train Epoch: 2 | Batch Status: 16000/50000 (32%) | Loss: 1.840580
Train Epoch: 2 | Batch Status: 16640/50000 (33%) | Loss: 1.747520
Train Epoch: 2 | Batch Status: 17280/50000 (35%) | Loss: 1.714726
Train Epoch: 2 | Batch Status: 17920/50000 (36%) | Loss: 1.896832
Train Epoch: 2 | Batch Status: 18560/50000 (37%) | Loss: 1.879629
Train Epoch: 2 | Batch Status: 19200/50000 (38%) | Loss: 1.770434
Train Epoch: 2 | Batch Status: 19840/50000 (40%) | Loss: 1.908672
Train Epoch: 2 | Batch Status: 20480/50000 (41%) | Loss: 1.826044
Train Epoch: 2 | Batch Status: 21120/50000 (42%) | Loss: 1.871686
Train Epoch: 2 | Batch Status: 21760/50000 (43%) | Loss: 1.979206
Train Epoch: 2 | Batch Status: 22400/50000 (45%) | Loss: 1.796172
Train Epoch: 2 | Batch Status: 23040/50000 (46%) | Loss: 1.795477
Train Epoch: 2 | Batch Status: 23680/50000 (47%) | Loss: 1.951971
Train Epoch: 2 | Batch Status: 24320/50000 (49%) | Loss: 1.581980
Train Epoch: 2 | Batch Status: 24960/50000 (50%) | Loss: 2.031019
Train Epoch: 2 | Batch Status: 25600/50000 (51%) | Loss: 1.850608
Train Epoch: 2 | Batch Status: 26240/50000 (52%) | Loss: 1.825027
Train Epoch: 2 | Batch Status: 26880/50000 (54%) | Loss: 1.916999
Train Epoch: 2 | Batch Status: 27520/50000 (55%) | Loss: 1.561738
Train Epoch: 2 | Batch Status: 28160/50000 (56%) | Loss: 1.713597
Train Epoch: 2 | Batch Status: 28800/50000 (58%) | Loss: 1.611829
Train Epoch: 2 | Batch Status: 29440/50000 (59%) | Loss: 1.724810
Train Epoch: 2 | Batch Status: 30080/50000 (60%) | Loss: 1.872650
Train Epoch: 2 | Batch Status: 30720/50000 (61%) | Loss: 1.631689
Train Epoch: 2 | Batch Status: 31360/50000 (63%) | Loss: 1.964037
Train Epoch: 2 | Batch Status: 32000/50000 (64%) | Loss: 1.923417
Train Epoch: 2 | Batch Status: 32640/50000 (65%) | Loss: 1.686870
Train Epoch: 2 | Batch Status: 33280/50000 (66%) | Loss: 1.807938
Train Epoch: 2 | Batch Status: 33920/50000 (68%) | Loss: 1.766989
Train Epoch: 2 | Batch Status: 34560/50000 (69%) | Loss: 1.843948
Train Epoch: 2 | Batch Status: 35200/50000 (70%) | Loss: 1.813942
Train Epoch: 2 | Batch Status: 35840/50000 (72%) | Loss: 1.867182
Train Epoch: 2 | Batch Status: 36480/50000 (73%) | Loss: 1.681229
Train Epoch: 2 | Batch Status: 37120/50000 (74%) | Loss: 1.863773
Train Epoch: 2 | Batch Status: 37760/50000 (75%) | Loss: 1.789447
Train Epoch: 2 | Batch Status: 38400/50000 (77%) | Loss: 1.851331
Train Epoch: 2 | Batch Status: 39040/50000 (78%) | Loss: 1.529489
Train Epoch: 2 | Batch Status: 39680/50000 (79%) | Loss: 2.009341
Train Epoch: 2 | Batch Status: 40320/50000 (81%) | Loss: 1.723275
Train Epoch: 2 | Batch Status: 40960/50000 (82%) | Loss: 1.541882
Train Epoch: 2 | Batch Status: 41600/50000 (83%) | Loss: 1.789372
Train Epoch: 2 | Batch Status: 42240/50000 (84%) | Loss: 1.717900
Train Epoch: 2 | Batch Status: 42880/50000 (86%) | Loss: 1.875676
Train Epoch: 2 | Batch Status: 43520/50000 (87%) | Loss: 1.776406
Train Epoch: 2 | Batch Status: 44160/50000 (88%) | Loss: 1.605618
Train Epoch: 2 | Batch Status: 44800/50000 (90%) | Loss: 2.000281
Train Epoch: 2 | Batch Status: 45440/50000 (91%) | Loss: 1.832736
Train Epoch: 2 | Batch Status: 46080/50000 (92%) | Loss: 1.568181
Train Epoch: 2 | Batch Status: 46720/50000 (93%) | Loss: 1.610326
Train Epoch: 2 | Batch Status: 47360/50000 (95%) | Loss: 1.755623
Train Epoch: 2 | Batch Status: 48000/50000 (96%) | Loss: 1.688871
Train Epoch: 2 | Batch Status: 48640/50000 (97%) | Loss: 1.789954
Train Epoch: 2 | Batch Status: 49280/50000 (98%) | Loss: 1.856670
Train Epoch: 2 | Batch Status: 49920/50000 (100%) | Loss: 1.922669
Training time: 0m 10s
===========================
Test set: Average loss: 0.0273, Accuracy: 3643/10000 (36%)
Testing time: 0m 12s
Train Epoch: 3 | Batch Status: 0/50000 (0%) | Loss: 1.762218
Train Epoch: 3 | Batch Status: 640/50000 (1%) | Loss: 1.594758
Train Epoch: 3 | Batch Status: 1280/50000 (3%) | Loss: 1.728581
Train Epoch: 3 | Batch Status: 1920/50000 (4%) | Loss: 1.738554
Train Epoch: 3 | Batch Status: 2560/50000 (5%) | Loss: 1.990599
Train Epoch: 3 | Batch Status: 3200/50000 (6%) | Loss: 1.792435
Train Epoch: 3 | Batch Status: 3840/50000 (8%) | Loss: 1.651345
Train Epoch: 3 | Batch Status: 4480/50000 (9%) | Loss: 1.731290
Train Epoch: 3 | Batch Status: 5120/50000 (10%) | Loss: 1.660680
Train Epoch: 3 | Batch Status: 5760/50000 (12%) | Loss: 1.736838
Train Epoch: 3 | Batch Status: 6400/50000 (13%) | Loss: 1.813382
Train Epoch: 3 | Batch Status: 7040/50000 (14%) | Loss: 1.732584
Train Epoch: 3 | Batch Status: 7680/50000 (15%) | Loss: 1.695634
Train Epoch: 3 | Batch Status: 8320/50000 (17%) | Loss: 1.626377
Train Epoch: 3 | Batch Status: 8960/50000 (18%) | Loss: 1.780159
Train Epoch: 3 | Batch Status: 9600/50000 (19%) | Loss: 1.633324
Train Epoch: 3 | Batch Status: 10240/50000 (20%) | Loss: 1.534077
Train Epoch: 3 | Batch Status: 10880/50000 (22%) | Loss: 1.692211
Train Epoch: 3 | Batch Status: 11520/50000 (23%) | Loss: 1.550120
Train Epoch: 3 | Batch Status: 12160/50000 (24%) | Loss: 1.680376
Train Epoch: 3 | Batch Status: 12800/50000 (26%) | Loss: 1.504925
Train Epoch: 3 | Batch Status: 13440/50000 (27%) | Loss: 1.700868
Train Epoch: 3 | Batch Status: 14080/50000 (28%) | Loss: 1.834452
Train Epoch: 3 | Batch Status: 14720/50000 (29%) | Loss: 1.694631
Train Epoch: 3 | Batch Status: 15360/50000 (31%) | Loss: 1.660741
Train Epoch: 3 | Batch Status: 16000/50000 (32%) | Loss: 1.865213
Train Epoch: 3 | Batch Status: 16640/50000 (33%) | Loss: 1.801937
Train Epoch: 3 | Batch Status: 17280/50000 (35%) | Loss: 1.876769
Train Epoch: 3 | Batch Status: 17920/50000 (36%) | Loss: 1.897163
Train Epoch: 3 | Batch Status: 18560/50000 (37%) | Loss: 1.839111
Train Epoch: 3 | Batch Status: 19200/50000 (38%) | Loss: 1.855641
Train Epoch: 3 | Batch Status: 19840/50000 (40%) | Loss: 1.730129
Train Epoch: 3 | Batch Status: 20480/50000 (41%) | Loss: 1.610934
Train Epoch: 3 | Batch Status: 21120/50000 (42%) | Loss: 1.810701
Train Epoch: 3 | Batch Status: 21760/50000 (43%) | Loss: 1.781146
Train Epoch: 3 | Batch Status: 22400/50000 (45%) | Loss: 1.940874
Train Epoch: 3 | Batch Status: 23040/50000 (46%) | Loss: 1.634106
Train Epoch: 3 | Batch Status: 23680/50000 (47%) | Loss: 1.830237
Train Epoch: 3 | Batch Status: 24320/50000 (49%) | Loss: 1.830707
Train Epoch: 3 | Batch Status: 24960/50000 (50%) | Loss: 1.774943
Train Epoch: 3 | Batch Status: 25600/50000 (51%) | Loss: 1.566643
Train Epoch: 3 | Batch Status: 26240/50000 (52%) | Loss: 1.652394
Train Epoch: 3 | Batch Status: 26880/50000 (54%) | Loss: 1.618044
Train Epoch: 3 | Batch Status: 27520/50000 (55%) | Loss: 1.755208
Train Epoch: 3 | Batch Status: 28160/50000 (56%) | Loss: 1.423275
Train Epoch: 3 | Batch Status: 28800/50000 (58%) | Loss: 1.707320
Train Epoch: 3 | Batch Status: 29440/50000 (59%) | Loss: 1.976113
Train Epoch: 3 | Batch Status: 30080/50000 (60%) | Loss: 1.719198
Train Epoch: 3 | Batch Status: 30720/50000 (61%) | Loss: 1.496974
Train Epoch: 3 | Batch Status: 31360/50000 (63%) | Loss: 1.603288
Train Epoch: 3 | Batch Status: 32000/50000 (64%) | Loss: 1.631134
Train Epoch: 3 | Batch Status: 32640/50000 (65%) | Loss: 1.753789
Train Epoch: 3 | Batch Status: 33280/50000 (66%) | Loss: 1.763349
Train Epoch: 3 | Batch Status: 33920/50000 (68%) | Loss: 1.603496
Train Epoch: 3 | Batch Status: 34560/50000 (69%) | Loss: 1.842609
Train Epoch: 3 | Batch Status: 35200/50000 (70%) | Loss: 1.567391
Train Epoch: 3 | Batch Status: 35840/50000 (72%) | Loss: 1.738204
Train Epoch: 3 | Batch Status: 36480/50000 (73%) | Loss: 1.466937
Train Epoch: 3 | Batch Status: 37120/50000 (74%) | Loss: 1.954135
Train Epoch: 3 | Batch Status: 37760/50000 (75%) | Loss: 1.638582
Train Epoch: 3 | Batch Status: 38400/50000 (77%) | Loss: 1.692541
Train Epoch: 3 | Batch Status: 39040/50000 (78%) | Loss: 1.497845
Train Epoch: 3 | Batch Status: 39680/50000 (79%) | Loss: 1.582831
Train Epoch: 3 | Batch Status: 40320/50000 (81%) | Loss: 1.736154
Train Epoch: 3 | Batch Status: 40960/50000 (82%) | Loss: 1.530252
Train Epoch: 3 | Batch Status: 41600/50000 (83%) | Loss: 1.488679
Train Epoch: 3 | Batch Status: 42240/50000 (84%) | Loss: 1.733930
Train Epoch: 3 | Batch Status: 42880/50000 (86%) | Loss: 1.794277
Train Epoch: 3 | Batch Status: 43520/50000 (87%) | Loss: 1.760163
Train Epoch: 3 | Batch Status: 44160/50000 (88%) | Loss: 1.754299
Train Epoch: 3 | Batch Status: 44800/50000 (90%) | Loss: 1.536210
Train Epoch: 3 | Batch Status: 45440/50000 (91%) | Loss: 1.690664
Train Epoch: 3 | Batch Status: 46080/50000 (92%) | Loss: 1.722892
Train Epoch: 3 | Batch Status: 46720/50000 (93%) | Loss: 1.792412
Train Epoch: 3 | Batch Status: 47360/50000 (95%) | Loss: 1.740867
Train Epoch: 3 | Batch Status: 48000/50000 (96%) | Loss: 1.577857
Train Epoch: 3 | Batch Status: 48640/50000 (97%) | Loss: 1.771557
Train Epoch: 3 | Batch Status: 49280/50000 (98%) | Loss: 1.818210
Train Epoch: 3 | Batch Status: 49920/50000 (100%) | Loss: 1.685427
Training time: 0m 10s
===========================
Test set: Average loss: 0.0278, Accuracy: 3755/10000 (38%)
Testing time: 0m 12s
Train Epoch: 4 | Batch Status: 0/50000 (0%) | Loss: 1.763024
Train Epoch: 4 | Batch Status: 640/50000 (1%) | Loss: 1.730238
Train Epoch: 4 | Batch Status: 1280/50000 (3%) | Loss: 1.510725
Train Epoch: 4 | Batch Status: 1920/50000 (4%) | Loss: 1.752685
Train Epoch: 4 | Batch Status: 2560/50000 (5%) | Loss: 1.708646
Train Epoch: 4 | Batch Status: 3200/50000 (6%) | Loss: 1.782881
Train Epoch: 4 | Batch Status: 3840/50000 (8%) | Loss: 1.448733
Train Epoch: 4 | Batch Status: 4480/50000 (9%) | Loss: 1.591520
Train Epoch: 4 | Batch Status: 5120/50000 (10%) | Loss: 1.475291
Train Epoch: 4 | Batch Status: 5760/50000 (12%) | Loss: 1.819665
Train Epoch: 4 | Batch Status: 6400/50000 (13%) | Loss: 1.452730
Train Epoch: 4 | Batch Status: 7040/50000 (14%) | Loss: 1.578565
Train Epoch: 4 | Batch Status: 7680/50000 (15%) | Loss: 1.562036
Train Epoch: 4 | Batch Status: 8320/50000 (17%) | Loss: 1.670693
Train Epoch: 4 | Batch Status: 8960/50000 (18%) | Loss: 1.766847
Train Epoch: 4 | Batch Status: 9600/50000 (19%) | Loss: 1.629829
Train Epoch: 4 | Batch Status: 10240/50000 (20%) | Loss: 1.761106
Train Epoch: 4 | Batch Status: 10880/50000 (22%) | Loss: 1.679489
Train Epoch: 4 | Batch Status: 11520/50000 (23%) | Loss: 2.002055
Train Epoch: 4 | Batch Status: 12160/50000 (24%) | Loss: 1.529874
Train Epoch: 4 | Batch Status: 12800/50000 (26%) | Loss: 1.487910
Train Epoch: 4 | Batch Status: 13440/50000 (27%) | Loss: 1.656164
Train Epoch: 4 | Batch Status: 14080/50000 (28%) | Loss: 1.551318
Train Epoch: 4 | Batch Status: 14720/50000 (29%) | Loss: 1.954861
Train Epoch: 4 | Batch Status: 15360/50000 (31%) | Loss: 1.768051
Train Epoch: 4 | Batch Status: 16000/50000 (32%) | Loss: 1.570283
Train Epoch: 4 | Batch Status: 16640/50000 (33%) | Loss: 1.707264
Train Epoch: 4 | Batch Status: 17280/50000 (35%) | Loss: 1.706153
Train Epoch: 4 | Batch Status: 17920/50000 (36%) | Loss: 1.649755
Train Epoch: 4 | Batch Status: 18560/50000 (37%) | Loss: 1.646014
Train Epoch: 4 | Batch Status: 19200/50000 (38%) | Loss: 1.579841
Train Epoch: 4 | Batch Status: 19840/50000 (40%) | Loss: 1.458999
Train Epoch: 4 | Batch Status: 20480/50000 (41%) | Loss: 1.827531
Train Epoch: 4 | Batch Status: 21120/50000 (42%) | Loss: 1.381345
Train Epoch: 4 | Batch Status: 21760/50000 (43%) | Loss: 1.566419
Train Epoch: 4 | Batch Status: 22400/50000 (45%) | Loss: 1.707339
Train Epoch: 4 | Batch Status: 23040/50000 (46%) | Loss: 1.698147
Train Epoch: 4 | Batch Status: 23680/50000 (47%) | Loss: 1.801255
Train Epoch: 4 | Batch Status: 24320/50000 (49%) | Loss: 1.577958
Train Epoch: 4 | Batch Status: 24960/50000 (50%) | Loss: 1.662395
Train Epoch: 4 | Batch Status: 25600/50000 (51%) | Loss: 1.679001
Train Epoch: 4 | Batch Status: 26240/50000 (52%) | Loss: 1.621643
Train Epoch: 4 | Batch Status: 26880/50000 (54%) | Loss: 1.757462
Train Epoch: 4 | Batch Status: 27520/50000 (55%) | Loss: 1.726611
Train Epoch: 4 | Batch Status: 28160/50000 (56%) | Loss: 1.541296
Train Epoch: 4 | Batch Status: 28800/50000 (58%) | Loss: 1.682536
Train Epoch: 4 | Batch Status: 29440/50000 (59%) | Loss: 1.572814
Train Epoch: 4 | Batch Status: 30080/50000 (60%) | Loss: 1.738388
Train Epoch: 4 | Batch Status: 30720/50000 (61%) | Loss: 1.629320
Train Epoch: 4 | Batch Status: 31360/50000 (63%) | Loss: 1.584896
Train Epoch: 4 | Batch Status: 32000/50000 (64%) | Loss: 1.669524
Train Epoch: 4 | Batch Status: 32640/50000 (65%) | Loss: 1.686173
Train Epoch: 4 | Batch Status: 33280/50000 (66%) | Loss: 1.597712
Train Epoch: 4 | Batch Status: 33920/50000 (68%) | Loss: 1.709437
Train Epoch: 4 | Batch Status: 34560/50000 (69%) | Loss: 1.736388
Train Epoch: 4 | Batch Status: 35200/50000 (70%) | Loss: 1.501469
Train Epoch: 4 | Batch Status: 35840/50000 (72%) | Loss: 1.645622
Train Epoch: 4 | Batch Status: 36480/50000 (73%) | Loss: 1.759071
Train Epoch: 4 | Batch Status: 37120/50000 (74%) | Loss: 1.751266
Train Epoch: 4 | Batch Status: 37760/50000 (75%) | Loss: 1.491060
Train Epoch: 4 | Batch Status: 38400/50000 (77%) | Loss: 1.574224
Train Epoch: 4 | Batch Status: 39040/50000 (78%) | Loss: 1.631950
Train Epoch: 4 | Batch Status: 39680/50000 (79%) | Loss: 1.641206
Train Epoch: 4 | Batch Status: 40320/50000 (81%) | Loss: 1.511662
Train Epoch: 4 | Batch Status: 40960/50000 (82%) | Loss: 1.686033
Train Epoch: 4 | Batch Status: 41600/50000 (83%) | Loss: 1.411848
Train Epoch: 4 | Batch Status: 42240/50000 (84%) | Loss: 1.688150
Train Epoch: 4 | Batch Status: 42880/50000 (86%) | Loss: 1.393399
Train Epoch: 4 | Batch Status: 43520/50000 (87%) | Loss: 1.635546
Train Epoch: 4 | Batch Status: 44160/50000 (88%) | Loss: 1.639409
Train Epoch: 4 | Batch Status: 44800/50000 (90%) | Loss: 1.610590
Train Epoch: 4 | Batch Status: 45440/50000 (91%) | Loss: 1.677259
Train Epoch: 4 | Batch Status: 46080/50000 (92%) | Loss: 1.491598
Train Epoch: 4 | Batch Status: 46720/50000 (93%) | Loss: 1.603878
Train Epoch: 4 | Batch Status: 47360/50000 (95%) | Loss: 1.702567
Train Epoch: 4 | Batch Status: 48000/50000 (96%) | Loss: 1.503626
Train Epoch: 4 | Batch Status: 48640/50000 (97%) | Loss: 1.551030
Train Epoch: 4 | Batch Status: 49280/50000 (98%) | Loss: 1.730313
Train Epoch: 4 | Batch Status: 49920/50000 (100%) | Loss: 1.513081
Training time: 0m 10s
===========================
Test set: Average loss: 0.0263, Accuracy: 3966/10000 (40%)
Testing time: 0m 12s
Train Epoch: 5 | Batch Status: 0/50000 (0%) | Loss: 1.722800
Train Epoch: 5 | Batch Status: 640/50000 (1%) | Loss: 1.621438
Train Epoch: 5 | Batch Status: 1280/50000 (3%) | Loss: 1.784843
Train Epoch: 5 | Batch Status: 1920/50000 (4%) | Loss: 1.525287
Train Epoch: 5 | Batch Status: 2560/50000 (5%) | Loss: 1.605141
Train Epoch: 5 | Batch Status: 3200/50000 (6%) | Loss: 1.527157
Train Epoch: 5 | Batch Status: 3840/50000 (8%) | Loss: 1.235672
Train Epoch: 5 | Batch Status: 4480/50000 (9%) | Loss: 1.381210
Train Epoch: 5 | Batch Status: 5120/50000 (10%) | Loss: 1.544140
Train Epoch: 5 | Batch Status: 5760/50000 (12%) | Loss: 1.582541
Train Epoch: 5 | Batch Status: 6400/50000 (13%) | Loss: 1.579264
Train Epoch: 5 | Batch Status: 7040/50000 (14%) | Loss: 1.299119
Train Epoch: 5 | Batch Status: 7680/50000 (15%) | Loss: 1.649429
Train Epoch: 5 | Batch Status: 8320/50000 (17%) | Loss: 1.676594
Train Epoch: 5 | Batch Status: 8960/50000 (18%) | Loss: 1.310101
Train Epoch: 5 | Batch Status: 9600/50000 (19%) | Loss: 1.564337
Train Epoch: 5 | Batch Status: 10240/50000 (20%) | Loss: 1.602390
Train Epoch: 5 | Batch Status: 10880/50000 (22%) | Loss: 1.576607
Train Epoch: 5 | Batch Status: 11520/50000 (23%) | Loss: 1.366529
Train Epoch: 5 | Batch Status: 12160/50000 (24%) | Loss: 1.536629
Train Epoch: 5 | Batch Status: 12800/50000 (26%) | Loss: 1.882386
Train Epoch: 5 | Batch Status: 13440/50000 (27%) | Loss: 1.507522
Train Epoch: 5 | Batch Status: 14080/50000 (28%) | Loss: 1.462965
Train Epoch: 5 | Batch Status: 14720/50000 (29%) | Loss: 1.583067
Train Epoch: 5 | Batch Status: 15360/50000 (31%) | Loss: 1.660143
Train Epoch: 5 | Batch Status: 16000/50000 (32%) | Loss: 1.404219
Train Epoch: 5 | Batch Status: 16640/50000 (33%) | Loss: 1.614201
Train Epoch: 5 | Batch Status: 17280/50000 (35%) | Loss: 1.512921
Train Epoch: 5 | Batch Status: 17920/50000 (36%) | Loss: 1.474630
Train Epoch: 5 | Batch Status: 18560/50000 (37%) | Loss: 1.616454
Train Epoch: 5 | Batch Status: 19200/50000 (38%) | Loss: 1.552421
Train Epoch: 5 | Batch Status: 19840/50000 (40%) | Loss: 1.573733
Train Epoch: 5 | Batch Status: 20480/50000 (41%) | Loss: 1.707561
Train Epoch: 5 | Batch Status: 21120/50000 (42%) | Loss: 1.707625
Train Epoch: 5 | Batch Status: 21760/50000 (43%) | Loss: 1.495822
Train Epoch: 5 | Batch Status: 22400/50000 (45%) | Loss: 1.659218
Train Epoch: 5 | Batch Status: 23040/50000 (46%) | Loss: 1.567790
Train Epoch: 5 | Batch Status: 23680/50000 (47%) | Loss: 1.468388
Train Epoch: 5 | Batch Status: 24320/50000 (49%) | Loss: 1.795810
Train Epoch: 5 | Batch Status: 24960/50000 (50%) | Loss: 1.511868
Train Epoch: 5 | Batch Status: 25600/50000 (51%) | Loss: 1.459409
Train Epoch: 5 | Batch Status: 26240/50000 (52%) | Loss: 1.938538
Train Epoch: 5 | Batch Status: 26880/50000 (54%) | Loss: 1.542972
Train Epoch: 5 | Batch Status: 27520/50000 (55%) | Loss: 1.780323
Train Epoch: 5 | Batch Status: 28160/50000 (56%) | Loss: 1.537169
Train Epoch: 5 | Batch Status: 28800/50000 (58%) | Loss: 1.660984
Train Epoch: 5 | Batch Status: 29440/50000 (59%) | Loss: 1.661369
Train Epoch: 5 | Batch Status: 30080/50000 (60%) | Loss: 1.631569
Train Epoch: 5 | Batch Status: 30720/50000 (61%) | Loss: 1.476000
Train Epoch: 5 | Batch Status: 31360/50000 (63%) | Loss: 1.539317
Train Epoch: 5 | Batch Status: 32000/50000 (64%) | Loss: 1.488831
Train Epoch: 5 | Batch Status: 32640/50000 (65%) | Loss: 1.639744
Train Epoch: 5 | Batch Status: 33280/50000 (66%) | Loss: 1.404167
Train Epoch: 5 | Batch Status: 33920/50000 (68%) | Loss: 1.432789
Train Epoch: 5 | Batch Status: 34560/50000 (69%) | Loss: 1.529997
Train Epoch: 5 | Batch Status: 35200/50000 (70%) | Loss: 1.684886
Train Epoch: 5 | Batch Status: 35840/50000 (72%) | Loss: 1.734177
Train Epoch: 5 | Batch Status: 36480/50000 (73%) | Loss: 1.486198
Train Epoch: 5 | Batch Status: 37120/50000 (74%) | Loss: 1.350344
Train Epoch: 5 | Batch Status: 37760/50000 (75%) | Loss: 1.510400
Train Epoch: 5 | Batch Status: 38400/50000 (77%) | Loss: 1.448227
Train Epoch: 5 | Batch Status: 39040/50000 (78%) | Loss: 1.499019
Train Epoch: 5 | Batch Status: 39680/50000 (79%) | Loss: 1.449178
Train Epoch: 5 | Batch Status: 40320/50000 (81%) | Loss: 1.678151
Train Epoch: 5 | Batch Status: 40960/50000 (82%) | Loss: 1.872582
Train Epoch: 5 | Batch Status: 41600/50000 (83%) | Loss: 1.577782
Train Epoch: 5 | Batch Status: 42240/50000 (84%) | Loss: 1.486836
Train Epoch: 5 | Batch Status: 42880/50000 (86%) | Loss: 1.755993
Train Epoch: 5 | Batch Status: 43520/50000 (87%) | Loss: 1.460205
Train Epoch: 5 | Batch Status: 44160/50000 (88%) | Loss: 1.647043
Train Epoch: 5 | Batch Status: 44800/50000 (90%) | Loss: 1.542748
Train Epoch: 5 | Batch Status: 45440/50000 (91%) | Loss: 1.469720
Train Epoch: 5 | Batch Status: 46080/50000 (92%) | Loss: 1.551015
Train Epoch: 5 | Batch Status: 46720/50000 (93%) | Loss: 1.668671
Train Epoch: 5 | Batch Status: 47360/50000 (95%) | Loss: 1.613285
Train Epoch: 5 | Batch Status: 48000/50000 (96%) | Loss: 1.430617
Train Epoch: 5 | Batch Status: 48640/50000 (97%) | Loss: 1.445614
Train Epoch: 5 | Batch Status: 49280/50000 (98%) | Loss: 1.556370
Train Epoch: 5 | Batch Status: 49920/50000 (100%) | Loss: 1.321801
Training time: 0m 10s
===========================
Test set: Average loss: 0.0241, Accuracy: 4477/10000 (45%)
Testing time: 0m 11s
Train Epoch: 6 | Batch Status: 0/50000 (0%) | Loss: 1.398678
Train Epoch: 6 | Batch Status: 640/50000 (1%) | Loss: 1.692341
Train Epoch: 6 | Batch Status: 1280/50000 (3%) | Loss: 1.641793
Train Epoch: 6 | Batch Status: 1920/50000 (4%) | Loss: 1.464204
Train Epoch: 6 | Batch Status: 2560/50000 (5%) | Loss: 1.411179
Train Epoch: 6 | Batch Status: 3200/50000 (6%) | Loss: 1.424902
Train Epoch: 6 | Batch Status: 3840/50000 (8%) | Loss: 1.547897
Train Epoch: 6 | Batch Status: 4480/50000 (9%) | Loss: 1.415113
Train Epoch: 6 | Batch Status: 5120/50000 (10%) | Loss: 1.637966
Train Epoch: 6 | Batch Status: 5760/50000 (12%) | Loss: 1.567324
Train Epoch: 6 | Batch Status: 6400/50000 (13%) | Loss: 1.356953
Train Epoch: 6 | Batch Status: 7040/50000 (14%) | Loss: 1.393218
Train Epoch: 6 | Batch Status: 7680/50000 (15%) | Loss: 1.868970
Train Epoch: 6 | Batch Status: 8320/50000 (17%) | Loss: 1.466660
Train Epoch: 6 | Batch Status: 8960/50000 (18%) | Loss: 1.527709
Train Epoch: 6 | Batch Status: 9600/50000 (19%) | Loss: 1.502406
Train Epoch: 6 | Batch Status: 10240/50000 (20%) | Loss: 1.670997
Train Epoch: 6 | Batch Status: 10880/50000 (22%) | Loss: 1.619766
Train Epoch: 6 | Batch Status: 11520/50000 (23%) | Loss: 1.472856
Train Epoch: 6 | Batch Status: 12160/50000 (24%) | Loss: 1.665820
Train Epoch: 6 | Batch Status: 12800/50000 (26%) | Loss: 1.365868
Train Epoch: 6 | Batch Status: 13440/50000 (27%) | Loss: 1.352750
Train Epoch: 6 | Batch Status: 14080/50000 (28%) | Loss: 1.327229
Train Epoch: 6 | Batch Status: 14720/50000 (29%) | Loss: 1.814789
Train Epoch: 6 | Batch Status: 15360/50000 (31%) | Loss: 1.481504
Train Epoch: 6 | Batch Status: 16000/50000 (32%) | Loss: 1.576722
Train Epoch: 6 | Batch Status: 16640/50000 (33%) | Loss: 1.537520
Train Epoch: 6 | Batch Status: 17280/50000 (35%) | Loss: 1.578455
Train Epoch: 6 | Batch Status: 17920/50000 (36%) | Loss: 1.558292
Train Epoch: 6 | Batch Status: 18560/50000 (37%) | Loss: 1.735784
Train Epoch: 6 | Batch Status: 19200/50000 (38%) | Loss: 1.662426
Train Epoch: 6 | Batch Status: 19840/50000 (40%) | Loss: 1.553964
Train Epoch: 6 | Batch Status: 20480/50000 (41%) | Loss: 1.336374
Train Epoch: 6 | Batch Status: 21120/50000 (42%) | Loss: 1.778234
Train Epoch: 6 | Batch Status: 21760/50000 (43%) | Loss: 1.579139
Train Epoch: 6 | Batch Status: 22400/50000 (45%) | Loss: 1.626710
Train Epoch: 6 | Batch Status: 23040/50000 (46%) | Loss: 1.440509
Train Epoch: 6 | Batch Status: 23680/50000 (47%) | Loss: 1.617506
Train Epoch: 6 | Batch Status: 24320/50000 (49%) | Loss: 1.579388
Train Epoch: 6 | Batch Status: 24960/50000 (50%) | Loss: 1.638276
Train Epoch: 6 | Batch Status: 25600/50000 (51%) | Loss: 1.373506
Train Epoch: 6 | Batch Status: 26240/50000 (52%) | Loss: 1.682038
Train Epoch: 6 | Batch Status: 26880/50000 (54%) | Loss: 1.623296
Train Epoch: 6 | Batch Status: 27520/50000 (55%) | Loss: 1.590022
Train Epoch: 6 | Batch Status: 28160/50000 (56%) | Loss: 1.703527
Train Epoch: 6 | Batch Status: 28800/50000 (58%) | Loss: 1.524330
Train Epoch: 6 | Batch Status: 29440/50000 (59%) | Loss: 1.425844
Train Epoch: 6 | Batch Status: 30080/50000 (60%) | Loss: 1.499088
Train Epoch: 6 | Batch Status: 30720/50000 (61%) | Loss: 1.840131
Train Epoch: 6 | Batch Status: 31360/50000 (63%) | Loss: 1.471959
Train Epoch: 6 | Batch Status: 32000/50000 (64%) | Loss: 1.252028
Train Epoch: 6 | Batch Status: 32640/50000 (65%) | Loss: 1.532451
Train Epoch: 6 | Batch Status: 33280/50000 (66%) | Loss: 1.560652
Train Epoch: 6 | Batch Status: 33920/50000 (68%) | Loss: 1.281284
Train Epoch: 6 | Batch Status: 34560/50000 (69%) | Loss: 1.680474
Train Epoch: 6 | Batch Status: 35200/50000 (70%) | Loss: 1.655475
Train Epoch: 6 | Batch Status: 35840/50000 (72%) | Loss: 1.573110
Train Epoch: 6 | Batch Status: 36480/50000 (73%) | Loss: 1.425976
Train Epoch: 6 | Batch Status: 37120/50000 (74%) | Loss: 1.272514
Train Epoch: 6 | Batch Status: 37760/50000 (75%) | Loss: 1.536918
Train Epoch: 6 | Batch Status: 38400/50000 (77%) | Loss: 1.455115
Train Epoch: 6 | Batch Status: 39040/50000 (78%) | Loss: 1.433576
Train Epoch: 6 | Batch Status: 39680/50000 (79%) | Loss: 1.613828
Train Epoch: 6 | Batch Status: 40320/50000 (81%) | Loss: 1.685464
Train Epoch: 6 | Batch Status: 40960/50000 (82%) | Loss: 1.490420
Train Epoch: 6 | Batch Status: 41600/50000 (83%) | Loss: 1.574229
Train Epoch: 6 | Batch Status: 42240/50000 (84%) | Loss: 1.499334
Train Epoch: 6 | Batch Status: 42880/50000 (86%) | Loss: 1.309880
Train Epoch: 6 | Batch Status: 43520/50000 (87%) | Loss: 1.351803
Train Epoch: 6 | Batch Status: 44160/50000 (88%) | Loss: 1.461302
Train Epoch: 6 | Batch Status: 44800/50000 (90%) | Loss: 1.557703
Train Epoch: 6 | Batch Status: 45440/50000 (91%) | Loss: 1.620410
Train Epoch: 6 | Batch Status: 46080/50000 (92%) | Loss: 1.705422
Train Epoch: 6 | Batch Status: 46720/50000 (93%) | Loss: 1.367133
Train Epoch: 6 | Batch Status: 47360/50000 (95%) | Loss: 1.384949
Train Epoch: 6 | Batch Status: 48000/50000 (96%) | Loss: 1.670471
Train Epoch: 6 | Batch Status: 48640/50000 (97%) | Loss: 1.383199
Train Epoch: 6 | Batch Status: 49280/50000 (98%) | Loss: 1.332912
Train Epoch: 6 | Batch Status: 49920/50000 (100%) | Loss: 1.550178
Training time: 0m 10s
===========================
Test set: Average loss: 0.0259, Accuracy: 4162/10000 (42%)
Testing time: 0m 12s
Train Epoch: 7 | Batch Status: 0/50000 (0%) | Loss: 1.557964
Train Epoch: 7 | Batch Status: 640/50000 (1%) | Loss: 1.496871
Train Epoch: 7 | Batch Status: 1280/50000 (3%) | Loss: 1.641567
Train Epoch: 7 | Batch Status: 1920/50000 (4%) | Loss: 1.601238
Train Epoch: 7 | Batch Status: 2560/50000 (5%) | Loss: 1.543888
Train Epoch: 7 | Batch Status: 3200/50000 (6%) | Loss: 1.414469
Train Epoch: 7 | Batch Status: 3840/50000 (8%) | Loss: 1.525786
Train Epoch: 7 | Batch Status: 4480/50000 (9%) | Loss: 1.090528
Train Epoch: 7 | Batch Status: 5120/50000 (10%) | Loss: 1.344386
Train Epoch: 7 | Batch Status: 5760/50000 (12%) | Loss: 1.538120
Train Epoch: 7 | Batch Status: 6400/50000 (13%) | Loss: 1.560194
Train Epoch: 7 | Batch Status: 7040/50000 (14%) | Loss: 1.567153
Train Epoch: 7 | Batch Status: 7680/50000 (15%) | Loss: 1.382098
Train Epoch: 7 | Batch Status: 8320/50000 (17%) | Loss: 1.508412
Train Epoch: 7 | Batch Status: 8960/50000 (18%) | Loss: 1.664150
Train Epoch: 7 | Batch Status: 9600/50000 (19%) | Loss: 1.547135
Train Epoch: 7 | Batch Status: 10240/50000 (20%) | Loss: 1.556248
Train Epoch: 7 | Batch Status: 10880/50000 (22%) | Loss: 1.463858
Train Epoch: 7 | Batch Status: 11520/50000 (23%) | Loss: 1.418038
Train Epoch: 7 | Batch Status: 12160/50000 (24%) | Loss: 1.424935
Train Epoch: 7 | Batch Status: 12800/50000 (26%) | Loss: 1.494167
Train Epoch: 7 | Batch Status: 13440/50000 (27%) | Loss: 1.701707
Train Epoch: 7 | Batch Status: 14080/50000 (28%) | Loss: 1.432119
Train Epoch: 7 | Batch Status: 14720/50000 (29%) | Loss: 1.053701
Train Epoch: 7 | Batch Status: 15360/50000 (31%) | Loss: 1.542255
Train Epoch: 7 | Batch Status: 16000/50000 (32%) | Loss: 1.469766
Train Epoch: 7 | Batch Status: 16640/50000 (33%) | Loss: 1.385843
Train Epoch: 7 | Batch Status: 17280/50000 (35%) | Loss: 1.668831
Train Epoch: 7 | Batch Status: 17920/50000 (36%) | Loss: 1.685154
Train Epoch: 7 | Batch Status: 18560/50000 (37%) | Loss: 1.345122
Train Epoch: 7 | Batch Status: 19200/50000 (38%) | Loss: 1.428328
Train Epoch: 7 | Batch Status: 19840/50000 (40%) | Loss: 1.445426
Train Epoch: 7 | Batch Status: 20480/50000 (41%) | Loss: 1.359038
Train Epoch: 7 | Batch Status: 21120/50000 (42%) | Loss: 1.206037
Train Epoch: 7 | Batch Status: 21760/50000 (43%) | Loss: 1.722020
Train Epoch: 7 | Batch Status: 22400/50000 (45%) | Loss: 1.275517
Train Epoch: 7 | Batch Status: 23040/50000 (46%) | Loss: 1.395431
Train Epoch: 7 | Batch Status: 23680/50000 (47%) | Loss: 1.321213
Train Epoch: 7 | Batch Status: 24320/50000 (49%) | Loss: 1.337212
Train Epoch: 7 | Batch Status: 24960/50000 (50%) | Loss: 1.530165
Train Epoch: 7 | Batch Status: 25600/50000 (51%) | Loss: 1.407302
Train Epoch: 7 | Batch Status: 26240/50000 (52%) | Loss: 1.684783
Train Epoch: 7 | Batch Status: 26880/50000 (54%) | Loss: 1.529991
Train Epoch: 7 | Batch Status: 27520/50000 (55%) | Loss: 1.275904
Train Epoch: 7 | Batch Status: 28160/50000 (56%) | Loss: 1.322466
Train Epoch: 7 | Batch Status: 28800/50000 (58%) | Loss: 1.386824
Train Epoch: 7 | Batch Status: 29440/50000 (59%) | Loss: 1.594810
Train Epoch: 7 | Batch Status: 30080/50000 (60%) | Loss: 1.367571
Train Epoch: 7 | Batch Status: 30720/50000 (61%) | Loss: 1.306660
Train Epoch: 7 | Batch Status: 31360/50000 (63%) | Loss: 1.301844
Train Epoch: 7 | Batch Status: 32000/50000 (64%) | Loss: 1.512773
Train Epoch: 7 | Batch Status: 32640/50000 (65%) | Loss: 1.214709
Train Epoch: 7 | Batch Status: 33280/50000 (66%) | Loss: 1.408032
Train Epoch: 7 | Batch Status: 33920/50000 (68%) | Loss: 1.545410
Train Epoch: 7 | Batch Status: 34560/50000 (69%) | Loss: 1.599195
Train Epoch: 7 | Batch Status: 35200/50000 (70%) | Loss: 1.462655
Train Epoch: 7 | Batch Status: 35840/50000 (72%) | Loss: 1.498680
Train Epoch: 7 | Batch Status: 36480/50000 (73%) | Loss: 1.573753
Train Epoch: 7 | Batch Status: 37120/50000 (74%) | Loss: 1.396249
Train Epoch: 7 | Batch Status: 37760/50000 (75%) | Loss: 1.403583
Train Epoch: 7 | Batch Status: 38400/50000 (77%) | Loss: 1.618924
Train Epoch: 7 | Batch Status: 39040/50000 (78%) | Loss: 1.499128
Train Epoch: 7 | Batch Status: 39680/50000 (79%) | Loss: 1.500325
Train Epoch: 7 | Batch Status: 40320/50000 (81%) | Loss: 1.478315
Train Epoch: 7 | Batch Status: 40960/50000 (82%) | Loss: 1.488090
Train Epoch: 7 | Batch Status: 41600/50000 (83%) | Loss: 1.257232
Train Epoch: 7 | Batch Status: 42240/50000 (84%) | Loss: 1.416896
Train Epoch: 7 | Batch Status: 42880/50000 (86%) | Loss: 1.167763
Train Epoch: 7 | Batch Status: 43520/50000 (87%) | Loss: 1.291266
Train Epoch: 7 | Batch Status: 44160/50000 (88%) | Loss: 1.567607
Train Epoch: 7 | Batch Status: 44800/50000 (90%) | Loss: 1.400465
Train Epoch: 7 | Batch Status: 45440/50000 (91%) | Loss: 1.485723
Train Epoch: 7 | Batch Status: 46080/50000 (92%) | Loss: 1.491140
Train Epoch: 7 | Batch Status: 46720/50000 (93%) | Loss: 1.478919
Train Epoch: 7 | Batch Status: 47360/50000 (95%) | Loss: 1.636417
Train Epoch: 7 | Batch Status: 48000/50000 (96%) | Loss: 1.513089
Train Epoch: 7 | Batch Status: 48640/50000 (97%) | Loss: 1.388139
Train Epoch: 7 | Batch Status: 49280/50000 (98%) | Loss: 1.288315
Train Epoch: 7 | Batch Status: 49920/50000 (100%) | Loss: 1.385871
Training time: 0m 10s
===========================
Test set: Average loss: 0.0256, Accuracy: 4359/10000 (44%)
Testing time: 0m 11s
Train Epoch: 8 | Batch Status: 0/50000 (0%) | Loss: 1.459292
Train Epoch: 8 | Batch Status: 640/50000 (1%) | Loss: 1.363846
Train Epoch: 8 | Batch Status: 1280/50000 (3%) | Loss: 1.615239
Train Epoch: 8 | Batch Status: 1920/50000 (4%) | Loss: 1.515270
Train Epoch: 8 | Batch Status: 2560/50000 (5%) | Loss: 1.468945
Train Epoch: 8 | Batch Status: 3200/50000 (6%) | Loss: 1.409978
Train Epoch: 8 | Batch Status: 3840/50000 (8%) | Loss: 1.485266
Train Epoch: 8 | Batch Status: 4480/50000 (9%) | Loss: 1.487283
Train Epoch: 8 | Batch Status: 5120/50000 (10%) | Loss: 1.451152
Train Epoch: 8 | Batch Status: 5760/50000 (12%) | Loss: 1.531110
Train Epoch: 8 | Batch Status: 6400/50000 (13%) | Loss: 1.446766
Train Epoch: 8 | Batch Status: 7040/50000 (14%) | Loss: 1.174850
Train Epoch: 8 | Batch Status: 7680/50000 (15%) | Loss: 1.394612
Train Epoch: 8 | Batch Status: 8320/50000 (17%) | Loss: 1.556606
Train Epoch: 8 | Batch Status: 8960/50000 (18%) | Loss: 1.683043
Train Epoch: 8 | Batch Status: 9600/50000 (19%) | Loss: 1.313228
Train Epoch: 8 | Batch Status: 10240/50000 (20%) | Loss: 1.463328
Train Epoch: 8 | Batch Status: 10880/50000 (22%) | Loss: 1.728016
Train Epoch: 8 | Batch Status: 11520/50000 (23%) | Loss: 1.341015
Train Epoch: 8 | Batch Status: 12160/50000 (24%) | Loss: 1.471039
Train Epoch: 8 | Batch Status: 12800/50000 (26%) | Loss: 1.312642
Train Epoch: 8 | Batch Status: 13440/50000 (27%) | Loss: 1.386271
Train Epoch: 8 | Batch Status: 14080/50000 (28%) | Loss: 1.405730
Train Epoch: 8 | Batch Status: 14720/50000 (29%) | Loss: 1.193670
Train Epoch: 8 | Batch Status: 15360/50000 (31%) | Loss: 1.472885
Train Epoch: 8 | Batch Status: 16000/50000 (32%) | Loss: 1.553362
Train Epoch: 8 | Batch Status: 16640/50000 (33%) | Loss: 1.459538
Train Epoch: 8 | Batch Status: 17280/50000 (35%) | Loss: 1.347272
Train Epoch: 8 | Batch Status: 17920/50000 (36%) | Loss: 1.311026
Train Epoch: 8 | Batch Status: 18560/50000 (37%) | Loss: 1.751199
Train Epoch: 8 | Batch Status: 19200/50000 (38%) | Loss: 1.393656
Train Epoch: 8 | Batch Status: 19840/50000 (40%) | Loss: 1.488407
Train Epoch: 8 | Batch Status: 20480/50000 (41%) | Loss: 1.409138
Train Epoch: 8 | Batch Status: 21120/50000 (42%) | Loss: 1.361278
Train Epoch: 8 | Batch Status: 21760/50000 (43%) | Loss: 1.326194
Train Epoch: 8 | Batch Status: 22400/50000 (45%) | Loss: 1.134929
Train Epoch: 8 | Batch Status: 23040/50000 (46%) | Loss: 1.569172
Train Epoch: 8 | Batch Status: 23680/50000 (47%) | Loss: 1.553026
Train Epoch: 8 | Batch Status: 24320/50000 (49%) | Loss: 1.328414
Train Epoch: 8 | Batch Status: 24960/50000 (50%) | Loss: 1.453615
Train Epoch: 8 | Batch Status: 25600/50000 (51%) | Loss: 1.405369
Train Epoch: 8 | Batch Status: 26240/50000 (52%) | Loss: 1.052603
Train Epoch: 8 | Batch Status: 26880/50000 (54%) | Loss: 1.516325
Train Epoch: 8 | Batch Status: 27520/50000 (55%) | Loss: 1.539222
Train Epoch: 8 | Batch Status: 28160/50000 (56%) | Loss: 1.437455
Train Epoch: 8 | Batch Status: 28800/50000 (58%) | Loss: 1.623194
Train Epoch: 8 | Batch Status: 29440/50000 (59%) | Loss: 1.469873
Train Epoch: 8 | Batch Status: 30080/50000 (60%) | Loss: 1.361832
Train Epoch: 8 | Batch Status: 30720/50000 (61%) | Loss: 1.267292
Train Epoch: 8 | Batch Status: 31360/50000 (63%) | Loss: 1.500578
Train Epoch: 8 | Batch Status: 32000/50000 (64%) | Loss: 1.467532
Train Epoch: 8 | Batch Status: 32640/50000 (65%) | Loss: 1.298883
Train Epoch: 8 | Batch Status: 33280/50000 (66%) | Loss: 1.475040
Train Epoch: 8 | Batch Status: 33920/50000 (68%) | Loss: 1.488526
Train Epoch: 8 | Batch Status: 34560/50000 (69%) | Loss: 1.432047
Train Epoch: 8 | Batch Status: 35200/50000 (70%) | Loss: 1.443959
Train Epoch: 8 | Batch Status: 35840/50000 (72%) | Loss: 1.396886
Train Epoch: 8 | Batch Status: 36480/50000 (73%) | Loss: 1.385373
Train Epoch: 8 | Batch Status: 37120/50000 (74%) | Loss: 1.400246
Train Epoch: 8 | Batch Status: 37760/50000 (75%) | Loss: 1.815891
Train Epoch: 8 | Batch Status: 38400/50000 (77%) | Loss: 1.486516
Train Epoch: 8 | Batch Status: 39040/50000 (78%) | Loss: 1.533774
Train Epoch: 8 | Batch Status: 39680/50000 (79%) | Loss: 1.691321
Train Epoch: 8 | Batch Status: 40320/50000 (81%) | Loss: 1.579463
Train Epoch: 8 | Batch Status: 40960/50000 (82%) | Loss: 1.493991
Train Epoch: 8 | Batch Status: 41600/50000 (83%) | Loss: 1.411912
Train Epoch: 8 | Batch Status: 42240/50000 (84%) | Loss: 1.518977
Train Epoch: 8 | Batch Status: 42880/50000 (86%) | Loss: 1.413389
Train Epoch: 8 | Batch Status: 43520/50000 (87%) | Loss: 1.475540
Train Epoch: 8 | Batch Status: 44160/50000 (88%) | Loss: 1.448334
Train Epoch: 8 | Batch Status: 44800/50000 (90%) | Loss: 1.651609
Train Epoch: 8 | Batch Status: 45440/50000 (91%) | Loss: 1.337728
Train Epoch: 8 | Batch Status: 46080/50000 (92%) | Loss: 1.524913
Train Epoch: 8 | Batch Status: 46720/50000 (93%) | Loss: 1.449340
Train Epoch: 8 | Batch Status: 47360/50000 (95%) | Loss: 1.320925
Train Epoch: 8 | Batch Status: 48000/50000 (96%) | Loss: 1.311678
Train Epoch: 8 | Batch Status: 48640/50000 (97%) | Loss: 1.552775
Train Epoch: 8 | Batch Status: 49280/50000 (98%) | Loss: 1.578990
Train Epoch: 8 | Batch Status: 49920/50000 (100%) | Loss: 1.102041
Training time: 0m 10s
===========================
Test set: Average loss: 0.0225, Accuracy: 4867/10000 (49%)
Testing time: 0m 12s
Train Epoch: 9 | Batch Status: 0/50000 (0%) | Loss: 1.295604
Train Epoch: 9 | Batch Status: 640/50000 (1%) | Loss: 1.422290
Train Epoch: 9 | Batch Status: 1280/50000 (3%) | Loss: 1.192046
Train Epoch: 9 | Batch Status: 1920/50000 (4%) | Loss: 1.270072
Train Epoch: 9 | Batch Status: 2560/50000 (5%) | Loss: 1.559921
Train Epoch: 9 | Batch Status: 3200/50000 (6%) | Loss: 1.318089
Train Epoch: 9 | Batch Status: 3840/50000 (8%) | Loss: 1.415810
Train Epoch: 9 | Batch Status: 4480/50000 (9%) | Loss: 1.106516
Train Epoch: 9 | Batch Status: 5120/50000 (10%) | Loss: 1.324204
Train Epoch: 9 | Batch Status: 5760/50000 (12%) | Loss: 1.392415
Train Epoch: 9 | Batch Status: 6400/50000 (13%) | Loss: 1.327364
Train Epoch: 9 | Batch Status: 7040/50000 (14%) | Loss: 1.448094
Train Epoch: 9 | Batch Status: 7680/50000 (15%) | Loss: 1.617276
Train Epoch: 9 | Batch Status: 8320/50000 (17%) | Loss: 1.280190
Train Epoch: 9 | Batch Status: 8960/50000 (18%) | Loss: 1.482141
Train Epoch: 9 | Batch Status: 9600/50000 (19%) | Loss: 1.576489
Train Epoch: 9 | Batch Status: 10240/50000 (20%) | Loss: 1.507393
Train Epoch: 9 | Batch Status: 10880/50000 (22%) | Loss: 1.324836
Train Epoch: 9 | Batch Status: 11520/50000 (23%) | Loss: 1.432677
Train Epoch: 9 | Batch Status: 12160/50000 (24%) | Loss: 1.374321
Train Epoch: 9 | Batch Status: 12800/50000 (26%) | Loss: 1.467826
Train Epoch: 9 | Batch Status: 13440/50000 (27%) | Loss: 1.181227
Train Epoch: 9 | Batch Status: 14080/50000 (28%) | Loss: 1.493345
Train Epoch: 9 | Batch Status: 14720/50000 (29%) | Loss: 1.315308
Train Epoch: 9 | Batch Status: 15360/50000 (31%) | Loss: 1.331175
Train Epoch: 9 | Batch Status: 16000/50000 (32%) | Loss: 1.430061
Train Epoch: 9 | Batch Status: 16640/50000 (33%) | Loss: 1.262127
Train Epoch: 9 | Batch Status: 17280/50000 (35%) | Loss: 1.212352
Train Epoch: 9 | Batch Status: 17920/50000 (36%) | Loss: 1.538783
Train Epoch: 9 | Batch Status: 18560/50000 (37%) | Loss: 1.403901
Train Epoch: 9 | Batch Status: 19200/50000 (38%) | Loss: 1.395501
Train Epoch: 9 | Batch Status: 19840/50000 (40%) | Loss: 1.405139
Train Epoch: 9 | Batch Status: 20480/50000 (41%) | Loss: 1.617482
Train Epoch: 9 | Batch Status: 21120/50000 (42%) | Loss: 1.444122
Train Epoch: 9 | Batch Status: 21760/50000 (43%) | Loss: 1.329338
Train Epoch: 9 | Batch Status: 22400/50000 (45%) | Loss: 1.048657
Train Epoch: 9 | Batch Status: 23040/50000 (46%) | Loss: 1.402407
Train Epoch: 9 | Batch Status: 23680/50000 (47%) | Loss: 1.399271
Train Epoch: 9 | Batch Status: 24320/50000 (49%) | Loss: 1.453615
Train Epoch: 9 | Batch Status: 24960/50000 (50%) | Loss: 1.286703
Train Epoch: 9 | Batch Status: 25600/50000 (51%) | Loss: 1.237334
Train Epoch: 9 | Batch Status: 26240/50000 (52%) | Loss: 1.398812
Train Epoch: 9 | Batch Status: 26880/50000 (54%) | Loss: 1.279628
Train Epoch: 9 | Batch Status: 27520/50000 (55%) | Loss: 1.402072
Train Epoch: 9 | Batch Status: 28160/50000 (56%) | Loss: 1.144561
Train Epoch: 9 | Batch Status: 28800/50000 (58%) | Loss: 1.361607
Train Epoch: 9 | Batch Status: 29440/50000 (59%) | Loss: 1.402572
Train Epoch: 9 | Batch Status: 30080/50000 (60%) | Loss: 1.422686
Train Epoch: 9 | Batch Status: 30720/50000 (61%) | Loss: 1.206221
Train Epoch: 9 | Batch Status: 31360/50000 (63%) | Loss: 1.451989
Train Epoch: 9 | Batch Status: 32000/50000 (64%) | Loss: 1.243414
Train Epoch: 9 | Batch Status: 32640/50000 (65%) | Loss: 1.060629
Train Epoch: 9 | Batch Status: 33280/50000 (66%) | Loss: 1.399269
Train Epoch: 9 | Batch Status: 33920/50000 (68%) | Loss: 1.458710
Train Epoch: 9 | Batch Status: 34560/50000 (69%) | Loss: 1.401175
Train Epoch: 9 | Batch Status: 35200/50000 (70%) | Loss: 1.357675
Train Epoch: 9 | Batch Status: 35840/50000 (72%) | Loss: 1.364030
Train Epoch: 9 | Batch Status: 36480/50000 (73%) | Loss: 1.460736
Train Epoch: 9 | Batch Status: 37120/50000 (74%) | Loss: 1.587523
Train Epoch: 9 | Batch Status: 37760/50000 (75%) | Loss: 1.636444
Train Epoch: 9 | Batch Status: 38400/50000 (77%) | Loss: 1.346685
Train Epoch: 9 | Batch Status: 39040/50000 (78%) | Loss: 1.536082
Train Epoch: 9 | Batch Status: 39680/50000 (79%) | Loss: 1.548629
Train Epoch: 9 | Batch Status: 40320/50000 (81%) | Loss: 1.455148
Train Epoch: 9 | Batch Status: 40960/50000 (82%) | Loss: 1.215310
Train Epoch: 9 | Batch Status: 41600/50000 (83%) | Loss: 1.377164
Train Epoch: 9 | Batch Status: 42240/50000 (84%) | Loss: 1.353452
Train Epoch: 9 | Batch Status: 42880/50000 (86%) | Loss: 1.349250
Train Epoch: 9 | Batch Status: 43520/50000 (87%) | Loss: 1.391396
Train Epoch: 9 | Batch Status: 44160/50000 (88%) | Loss: 1.322680
Train Epoch: 9 | Batch Status: 44800/50000 (90%) | Loss: 1.282531
Train Epoch: 9 | Batch Status: 45440/50000 (91%) | Loss: 1.351766
Train Epoch: 9 | Batch Status: 46080/50000 (92%) | Loss: 1.384514
Train Epoch: 9 | Batch Status: 46720/50000 (93%) | Loss: 1.400617
Train Epoch: 9 | Batch Status: 47360/50000 (95%) | Loss: 1.244425
Train Epoch: 9 | Batch Status: 48000/50000 (96%) | Loss: 1.514940
Train Epoch: 9 | Batch Status: 48640/50000 (97%) | Loss: 1.370787
Train Epoch: 9 | Batch Status: 49280/50000 (98%) | Loss: 0.961758
Train Epoch: 9 | Batch Status: 49920/50000 (100%) | Loss: 1.448460
Training time: 0m 10s
===========================
Test set: Average loss: 0.0226, Accuracy: 4891/10000 (49%)
Testing time: 0m 11s
Total Time: 1m 44s
Model was trained on cuda!
'''
