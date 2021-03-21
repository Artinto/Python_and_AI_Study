from __future__ import print_function
from torch import nn, optim, cuda
from torch.utils import data
from torchvision import datasets, transforms
import torch.nn.functional as F
import time

# Training settings
batch_size = 64
device = 'cuda' if cuda.is_available() else 'cpu'
print(f'Training MNIST Model on {device}\n{"=" * 44}')

# MNIST Dataset
#CIFAR-10 Dataset
train_dataset = datasets.CIFAR10(root='./data', train=True,download=True, transform=transforms.ToTensor())
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor())
#Data Loader
train_loader=data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader=data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.l1=nn.Linear(32*32*3,2000)
    self.l2=nn.Linear(2000, 1500)
    self.l3=nn.Linear(1500, 1000)
    self.l4=nn.Linear(1000, 500)
    self.l5=nn.Linear(500, 120)
    self.l6=nn.Linear(120, 10)
    

  def forward(self, x):
    x=x.view(-1, 32*32*3)#일자로 피는 작업
    x=F.relu(self.l1(x))
    x=F.relu(self.l2(x))
    x=F.relu(self.l3(x))
    x=F.relu(self.l4(x))
    x=F.relu(self.l5(x))
    return self.l6(x)


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
    
    Training MNIST Model on cuda
============================================
Files already downloaded and verified
Train Epoch: 1 | Batch Status: 0/50000 (0%) | Loss: 2.295937
Train Epoch: 1 | Batch Status: 640/50000 (1%) | Loss: 2.304573
Train Epoch: 1 | Batch Status: 1280/50000 (3%) | Loss: 2.304394
Train Epoch: 1 | Batch Status: 1920/50000 (4%) | Loss: 2.301201
Train Epoch: 1 | Batch Status: 2560/50000 (5%) | Loss: 2.297442
Train Epoch: 1 | Batch Status: 3200/50000 (6%) | Loss: 2.308009
Train Epoch: 1 | Batch Status: 3840/50000 (8%) | Loss: 2.302255
Train Epoch: 1 | Batch Status: 4480/50000 (9%) | Loss: 2.308537
Train Epoch: 1 | Batch Status: 5120/50000 (10%) | Loss: 2.290956
Train Epoch: 1 | Batch Status: 5760/50000 (12%) | Loss: 2.291892
Train Epoch: 1 | Batch Status: 6400/50000 (13%) | Loss: 2.285816
Train Epoch: 1 | Batch Status: 7040/50000 (14%) | Loss: 2.291921
Train Epoch: 1 | Batch Status: 7680/50000 (15%) | Loss: 2.279419
Train Epoch: 1 | Batch Status: 8320/50000 (17%) | Loss: 2.281067
Train Epoch: 1 | Batch Status: 8960/50000 (18%) | Loss: 2.282973
Train Epoch: 1 | Batch Status: 9600/50000 (19%) | Loss: 2.254894
Train Epoch: 1 | Batch Status: 10240/50000 (20%) | Loss: 2.240301
Train Epoch: 1 | Batch Status: 10880/50000 (22%) | Loss: 2.184160
Train Epoch: 1 | Batch Status: 11520/50000 (23%) | Loss: 2.176125
Train Epoch: 1 | Batch Status: 12160/50000 (24%) | Loss: 2.221303
Train Epoch: 1 | Batch Status: 12800/50000 (26%) | Loss: 2.073978
Train Epoch: 1 | Batch Status: 13440/50000 (27%) | Loss: 2.258576
Train Epoch: 1 | Batch Status: 14080/50000 (28%) | Loss: 2.108767
Train Epoch: 1 | Batch Status: 14720/50000 (29%) | Loss: 2.254076
Train Epoch: 1 | Batch Status: 15360/50000 (31%) | Loss: 2.013675
Train Epoch: 1 | Batch Status: 16000/50000 (32%) | Loss: 2.088094
Train Epoch: 1 | Batch Status: 16640/50000 (33%) | Loss: 2.109238
Train Epoch: 1 | Batch Status: 17280/50000 (35%) | Loss: 2.101111
Train Epoch: 1 | Batch Status: 17920/50000 (36%) | Loss: 2.231123
Train Epoch: 1 | Batch Status: 18560/50000 (37%) | Loss: 2.053238
Train Epoch: 1 | Batch Status: 19200/50000 (38%) | Loss: 2.002659
Train Epoch: 1 | Batch Status: 19840/50000 (40%) | Loss: 1.908087
Train Epoch: 1 | Batch Status: 20480/50000 (41%) | Loss: 2.060155
Train Epoch: 1 | Batch Status: 21120/50000 (42%) | Loss: 2.037530
Train Epoch: 1 | Batch Status: 21760/50000 (43%) | Loss: 2.116450
Train Epoch: 1 | Batch Status: 22400/50000 (45%) | Loss: 2.076522
Train Epoch: 1 | Batch Status: 23040/50000 (46%) | Loss: 2.145305
Train Epoch: 1 | Batch Status: 23680/50000 (47%) | Loss: 1.991516
Train Epoch: 1 | Batch Status: 24320/50000 (49%) | Loss: 2.003251
Train Epoch: 1 | Batch Status: 24960/50000 (50%) | Loss: 2.129791
Train Epoch: 1 | Batch Status: 25600/50000 (51%) | Loss: 2.068260
Train Epoch: 1 | Batch Status: 26240/50000 (52%) | Loss: 2.124354
Train Epoch: 1 | Batch Status: 26880/50000 (54%) | Loss: 1.975779
Train Epoch: 1 | Batch Status: 27520/50000 (55%) | Loss: 1.983280
Train Epoch: 1 | Batch Status: 28160/50000 (56%) | Loss: 1.954173
Train Epoch: 1 | Batch Status: 28800/50000 (58%) | Loss: 1.925455
Train Epoch: 1 | Batch Status: 29440/50000 (59%) | Loss: 1.963508
Train Epoch: 1 | Batch Status: 30080/50000 (60%) | Loss: 2.001031
Train Epoch: 1 | Batch Status: 30720/50000 (61%) | Loss: 1.980942
Train Epoch: 1 | Batch Status: 31360/50000 (63%) | Loss: 2.103566
Train Epoch: 1 | Batch Status: 32000/50000 (64%) | Loss: 1.914090
Train Epoch: 1 | Batch Status: 32640/50000 (65%) | Loss: 2.125587
Train Epoch: 1 | Batch Status: 33280/50000 (66%) | Loss: 2.007154
Train Epoch: 1 | Batch Status: 33920/50000 (68%) | Loss: 2.209452
Train Epoch: 1 | Batch Status: 34560/50000 (69%) | Loss: 1.898438
Train Epoch: 1 | Batch Status: 35200/50000 (70%) | Loss: 2.033656
Train Epoch: 1 | Batch Status: 35840/50000 (72%) | Loss: 1.889912
Train Epoch: 1 | Batch Status: 36480/50000 (73%) | Loss: 1.950605
Train Epoch: 1 | Batch Status: 37120/50000 (74%) | Loss: 1.960443
Train Epoch: 1 | Batch Status: 37760/50000 (75%) | Loss: 2.058729
Train Epoch: 1 | Batch Status: 38400/50000 (77%) | Loss: 1.906685
Train Epoch: 1 | Batch Status: 39040/50000 (78%) | Loss: 1.917408
Train Epoch: 1 | Batch Status: 39680/50000 (79%) | Loss: 2.025995
Train Epoch: 1 | Batch Status: 40320/50000 (81%) | Loss: 1.951194
Train Epoch: 1 | Batch Status: 40960/50000 (82%) | Loss: 1.952886
Train Epoch: 1 | Batch Status: 41600/50000 (83%) | Loss: 2.029428
Train Epoch: 1 | Batch Status: 42240/50000 (84%) | Loss: 2.062573
Train Epoch: 1 | Batch Status: 42880/50000 (86%) | Loss: 1.941503
Train Epoch: 1 | Batch Status: 43520/50000 (87%) | Loss: 1.861098
Train Epoch: 1 | Batch Status: 44160/50000 (88%) | Loss: 1.787896
Train Epoch: 1 | Batch Status: 44800/50000 (90%) | Loss: 1.998647
Train Epoch: 1 | Batch Status: 45440/50000 (91%) | Loss: 1.837980
Train Epoch: 1 | Batch Status: 46080/50000 (92%) | Loss: 1.943634
Train Epoch: 1 | Batch Status: 46720/50000 (93%) | Loss: 1.742723
Train Epoch: 1 | Batch Status: 47360/50000 (95%) | Loss: 1.814779
Train Epoch: 1 | Batch Status: 48000/50000 (96%) | Loss: 1.946549
Train Epoch: 1 | Batch Status: 48640/50000 (97%) | Loss: 2.007648
Train Epoch: 1 | Batch Status: 49280/50000 (98%) | Loss: 2.003040
Train Epoch: 1 | Batch Status: 49920/50000 (100%) | Loss: 1.839218
Training time: 0m 6s
===========================
Test set: Average loss: 0.0301, Accuracy: 2813/10000 (28%)
Testing time: 0m 7s
Train Epoch: 2 | Batch Status: 0/50000 (0%) | Loss: 1.786143
Train Epoch: 2 | Batch Status: 640/50000 (1%) | Loss: 1.691067
Train Epoch: 2 | Batch Status: 1280/50000 (3%) | Loss: 1.876029
Train Epoch: 2 | Batch Status: 1920/50000 (4%) | Loss: 1.850481
Train Epoch: 2 | Batch Status: 2560/50000 (5%) | Loss: 1.823163
Train Epoch: 2 | Batch Status: 3200/50000 (6%) | Loss: 1.933511
Train Epoch: 2 | Batch Status: 3840/50000 (8%) | Loss: 1.755996
Train Epoch: 2 | Batch Status: 4480/50000 (9%) | Loss: 1.898744
Train Epoch: 2 | Batch Status: 5120/50000 (10%) | Loss: 1.884216
Train Epoch: 2 | Batch Status: 5760/50000 (12%) | Loss: 1.927127
Train Epoch: 2 | Batch Status: 6400/50000 (13%) | Loss: 1.937777
Train Epoch: 2 | Batch Status: 7040/50000 (14%) | Loss: 1.652728
Train Epoch: 2 | Batch Status: 7680/50000 (15%) | Loss: 1.669386
Train Epoch: 2 | Batch Status: 8320/50000 (17%) | Loss: 1.786688
Train Epoch: 2 | Batch Status: 8960/50000 (18%) | Loss: 1.668981
Train Epoch: 2 | Batch Status: 9600/50000 (19%) | Loss: 1.925884
Train Epoch: 2 | Batch Status: 10240/50000 (20%) | Loss: 1.792834
Train Epoch: 2 | Batch Status: 10880/50000 (22%) | Loss: 1.735271
Train Epoch: 2 | Batch Status: 11520/50000 (23%) | Loss: 1.702786
Train Epoch: 2 | Batch Status: 12160/50000 (24%) | Loss: 1.799980
Train Epoch: 2 | Batch Status: 12800/50000 (26%) | Loss: 1.769640
Train Epoch: 2 | Batch Status: 13440/50000 (27%) | Loss: 1.874863
Train Epoch: 2 | Batch Status: 14080/50000 (28%) | Loss: 2.058511
Train Epoch: 2 | Batch Status: 14720/50000 (29%) | Loss: 1.836388
Train Epoch: 2 | Batch Status: 15360/50000 (31%) | Loss: 1.715126
Train Epoch: 2 | Batch Status: 16000/50000 (32%) | Loss: 1.817713
Train Epoch: 2 | Batch Status: 16640/50000 (33%) | Loss: 1.879277
Train Epoch: 2 | Batch Status: 17280/50000 (35%) | Loss: 1.680417
Train Epoch: 2 | Batch Status: 17920/50000 (36%) | Loss: 1.747277
Train Epoch: 2 | Batch Status: 18560/50000 (37%) | Loss: 1.629851
Train Epoch: 2 | Batch Status: 19200/50000 (38%) | Loss: 1.906685
Train Epoch: 2 | Batch Status: 19840/50000 (40%) | Loss: 1.921296
Train Epoch: 2 | Batch Status: 20480/50000 (41%) | Loss: 1.705249
Train Epoch: 2 | Batch Status: 21120/50000 (42%) | Loss: 1.778125
Train Epoch: 2 | Batch Status: 21760/50000 (43%) | Loss: 1.688431
Train Epoch: 2 | Batch Status: 22400/50000 (45%) | Loss: 1.875061
Train Epoch: 2 | Batch Status: 23040/50000 (46%) | Loss: 1.634662
Train Epoch: 2 | Batch Status: 23680/50000 (47%) | Loss: 2.015603
Train Epoch: 2 | Batch Status: 24320/50000 (49%) | Loss: 1.744933
Train Epoch: 2 | Batch Status: 24960/50000 (50%) | Loss: 1.948802
Train Epoch: 2 | Batch Status: 25600/50000 (51%) | Loss: 1.551072
Train Epoch: 2 | Batch Status: 26240/50000 (52%) | Loss: 1.645061
Train Epoch: 2 | Batch Status: 26880/50000 (54%) | Loss: 1.719152
Train Epoch: 2 | Batch Status: 27520/50000 (55%) | Loss: 1.771975
Train Epoch: 2 | Batch Status: 28160/50000 (56%) | Loss: 2.068738
Train Epoch: 2 | Batch Status: 28800/50000 (58%) | Loss: 1.953180
Train Epoch: 2 | Batch Status: 29440/50000 (59%) | Loss: 1.972718
Train Epoch: 2 | Batch Status: 30080/50000 (60%) | Loss: 1.702310
Train Epoch: 2 | Batch Status: 30720/50000 (61%) | Loss: 1.829710
Train Epoch: 2 | Batch Status: 31360/50000 (63%) | Loss: 1.646725
Train Epoch: 2 | Batch Status: 32000/50000 (64%) | Loss: 1.819684
Train Epoch: 2 | Batch Status: 32640/50000 (65%) | Loss: 1.799228
Train Epoch: 2 | Batch Status: 33280/50000 (66%) | Loss: 1.725226
Train Epoch: 2 | Batch Status: 33920/50000 (68%) | Loss: 1.870337
Train Epoch: 2 | Batch Status: 34560/50000 (69%) | Loss: 1.921127
Train Epoch: 2 | Batch Status: 35200/50000 (70%) | Loss: 1.743172
Train Epoch: 2 | Batch Status: 35840/50000 (72%) | Loss: 1.647043
Train Epoch: 2 | Batch Status: 36480/50000 (73%) | Loss: 1.853842
Train Epoch: 2 | Batch Status: 37120/50000 (74%) | Loss: 1.741986
Train Epoch: 2 | Batch Status: 37760/50000 (75%) | Loss: 1.751272
Train Epoch: 2 | Batch Status: 38400/50000 (77%) | Loss: 1.884568
Train Epoch: 2 | Batch Status: 39040/50000 (78%) | Loss: 1.827108
Train Epoch: 2 | Batch Status: 39680/50000 (79%) | Loss: 1.886523
Train Epoch: 2 | Batch Status: 40320/50000 (81%) | Loss: 1.970567
Train Epoch: 2 | Batch Status: 40960/50000 (82%) | Loss: 1.742550
Train Epoch: 2 | Batch Status: 41600/50000 (83%) | Loss: 1.728346
Train Epoch: 2 | Batch Status: 42240/50000 (84%) | Loss: 2.052968
Train Epoch: 2 | Batch Status: 42880/50000 (86%) | Loss: 1.842915
Train Epoch: 2 | Batch Status: 43520/50000 (87%) | Loss: 1.747396
Train Epoch: 2 | Batch Status: 44160/50000 (88%) | Loss: 1.876682
Train Epoch: 2 | Batch Status: 44800/50000 (90%) | Loss: 1.627827
Train Epoch: 2 | Batch Status: 45440/50000 (91%) | Loss: 1.597157
Train Epoch: 2 | Batch Status: 46080/50000 (92%) | Loss: 1.563143
Train Epoch: 2 | Batch Status: 46720/50000 (93%) | Loss: 1.759719
Train Epoch: 2 | Batch Status: 47360/50000 (95%) | Loss: 1.680360
Train Epoch: 2 | Batch Status: 48000/50000 (96%) | Loss: 1.710054
Train Epoch: 2 | Batch Status: 48640/50000 (97%) | Loss: 1.769175
Train Epoch: 2 | Batch Status: 49280/50000 (98%) | Loss: 1.616062
Train Epoch: 2 | Batch Status: 49920/50000 (100%) | Loss: 1.607797
Training time: 0m 6s
===========================
Test set: Average loss: 0.0286, Accuracy: 3436/10000 (34%)
Testing time: 0m 7s
Train Epoch: 3 | Batch Status: 0/50000 (0%) | Loss: 1.830353
Train Epoch: 3 | Batch Status: 640/50000 (1%) | Loss: 1.646334
Train Epoch: 3 | Batch Status: 1280/50000 (3%) | Loss: 1.477382
Train Epoch: 3 | Batch Status: 1920/50000 (4%) | Loss: 1.737634
Train Epoch: 3 | Batch Status: 2560/50000 (5%) | Loss: 1.802261
Train Epoch: 3 | Batch Status: 3200/50000 (6%) | Loss: 1.705538
Train Epoch: 3 | Batch Status: 3840/50000 (8%) | Loss: 1.729925
Train Epoch: 3 | Batch Status: 4480/50000 (9%) | Loss: 1.711529
Train Epoch: 3 | Batch Status: 5120/50000 (10%) | Loss: 1.731821
Train Epoch: 3 | Batch Status: 5760/50000 (12%) | Loss: 1.622165
Train Epoch: 3 | Batch Status: 6400/50000 (13%) | Loss: 1.945480
Train Epoch: 3 | Batch Status: 7040/50000 (14%) | Loss: 1.533885
Train Epoch: 3 | Batch Status: 7680/50000 (15%) | Loss: 1.719096
Train Epoch: 3 | Batch Status: 8320/50000 (17%) | Loss: 1.644725
Train Epoch: 3 | Batch Status: 8960/50000 (18%) | Loss: 1.712538
Train Epoch: 3 | Batch Status: 9600/50000 (19%) | Loss: 1.667338
Train Epoch: 3 | Batch Status: 10240/50000 (20%) | Loss: 1.783536
Train Epoch: 3 | Batch Status: 10880/50000 (22%) | Loss: 1.460358
Train Epoch: 3 | Batch Status: 11520/50000 (23%) | Loss: 1.604814
Train Epoch: 3 | Batch Status: 12160/50000 (24%) | Loss: 1.688219
Train Epoch: 3 | Batch Status: 12800/50000 (26%) | Loss: 1.548685
Train Epoch: 3 | Batch Status: 13440/50000 (27%) | Loss: 1.716594
Train Epoch: 3 | Batch Status: 14080/50000 (28%) | Loss: 2.017993
Train Epoch: 3 | Batch Status: 14720/50000 (29%) | Loss: 1.782233
Train Epoch: 3 | Batch Status: 15360/50000 (31%) | Loss: 1.774645
Train Epoch: 3 | Batch Status: 16000/50000 (32%) | Loss: 1.869370
Train Epoch: 3 | Batch Status: 16640/50000 (33%) | Loss: 1.544949
Train Epoch: 3 | Batch Status: 17280/50000 (35%) | Loss: 1.732660
Train Epoch: 3 | Batch Status: 17920/50000 (36%) | Loss: 1.725714
Train Epoch: 3 | Batch Status: 18560/50000 (37%) | Loss: 1.736917
Train Epoch: 3 | Batch Status: 19200/50000 (38%) | Loss: 1.519778
Train Epoch: 3 | Batch Status: 19840/50000 (40%) | Loss: 1.800427
Train Epoch: 3 | Batch Status: 20480/50000 (41%) | Loss: 1.604942
Train Epoch: 3 | Batch Status: 21120/50000 (42%) | Loss: 1.580620
Train Epoch: 3 | Batch Status: 21760/50000 (43%) | Loss: 1.709129
Train Epoch: 3 | Batch Status: 22400/50000 (45%) | Loss: 1.560353
Train Epoch: 3 | Batch Status: 23040/50000 (46%) | Loss: 1.808094
Train Epoch: 3 | Batch Status: 23680/50000 (47%) | Loss: 1.934644
Train Epoch: 3 | Batch Status: 24320/50000 (49%) | Loss: 1.817200
Train Epoch: 3 | Batch Status: 24960/50000 (50%) | Loss: 1.728616
Train Epoch: 3 | Batch Status: 25600/50000 (51%) | Loss: 1.773985
Train Epoch: 3 | Batch Status: 26240/50000 (52%) | Loss: 1.632298
Train Epoch: 3 | Batch Status: 26880/50000 (54%) | Loss: 1.504094
Train Epoch: 3 | Batch Status: 27520/50000 (55%) | Loss: 1.686297
Train Epoch: 3 | Batch Status: 28160/50000 (56%) | Loss: 1.413431
Train Epoch: 3 | Batch Status: 28800/50000 (58%) | Loss: 1.593794
Train Epoch: 3 | Batch Status: 29440/50000 (59%) | Loss: 1.772425
Train Epoch: 3 | Batch Status: 30080/50000 (60%) | Loss: 1.796829
Train Epoch: 3 | Batch Status: 30720/50000 (61%) | Loss: 1.651033
Train Epoch: 3 | Batch Status: 31360/50000 (63%) | Loss: 1.635146
Train Epoch: 3 | Batch Status: 32000/50000 (64%) | Loss: 1.688290
Train Epoch: 3 | Batch Status: 32640/50000 (65%) | Loss: 1.528226
Train Epoch: 3 | Batch Status: 33280/50000 (66%) | Loss: 1.833571
Train Epoch: 3 | Batch Status: 33920/50000 (68%) | Loss: 1.604950
Train Epoch: 3 | Batch Status: 34560/50000 (69%) | Loss: 1.660785
Train Epoch: 3 | Batch Status: 35200/50000 (70%) | Loss: 1.631970
Train Epoch: 3 | Batch Status: 35840/50000 (72%) | Loss: 1.854113
Train Epoch: 3 | Batch Status: 36480/50000 (73%) | Loss: 1.837040
Train Epoch: 3 | Batch Status: 37120/50000 (74%) | Loss: 1.610322
Train Epoch: 3 | Batch Status: 37760/50000 (75%) | Loss: 1.493249
Train Epoch: 3 | Batch Status: 38400/50000 (77%) | Loss: 1.461360
Train Epoch: 3 | Batch Status: 39040/50000 (78%) | Loss: 1.703177
Train Epoch: 3 | Batch Status: 39680/50000 (79%) | Loss: 1.510061
Train Epoch: 3 | Batch Status: 40320/50000 (81%) | Loss: 1.622871
Train Epoch: 3 | Batch Status: 40960/50000 (82%) | Loss: 1.642298
Train Epoch: 3 | Batch Status: 41600/50000 (83%) | Loss: 1.796654
Train Epoch: 3 | Batch Status: 42240/50000 (84%) | Loss: 1.793797
Train Epoch: 3 | Batch Status: 42880/50000 (86%) | Loss: 1.811093
Train Epoch: 3 | Batch Status: 43520/50000 (87%) | Loss: 1.468057
Train Epoch: 3 | Batch Status: 44160/50000 (88%) | Loss: 1.723176
Train Epoch: 3 | Batch Status: 44800/50000 (90%) | Loss: 1.759157
Train Epoch: 3 | Batch Status: 45440/50000 (91%) | Loss: 1.702563
Train Epoch: 3 | Batch Status: 46080/50000 (92%) | Loss: 1.561290
Train Epoch: 3 | Batch Status: 46720/50000 (93%) | Loss: 1.694695
Train Epoch: 3 | Batch Status: 47360/50000 (95%) | Loss: 1.646491
Train Epoch: 3 | Batch Status: 48000/50000 (96%) | Loss: 1.529766
Train Epoch: 3 | Batch Status: 48640/50000 (97%) | Loss: 1.831504
Train Epoch: 3 | Batch Status: 49280/50000 (98%) | Loss: 1.525713
Train Epoch: 3 | Batch Status: 49920/50000 (100%) | Loss: 1.518073
Training time: 0m 6s
===========================
Test set: Average loss: 0.0261, Accuracy: 4087/10000 (41%)
Testing time: 0m 7s
Train Epoch: 4 | Batch Status: 0/50000 (0%) | Loss: 1.500043
Train Epoch: 4 | Batch Status: 640/50000 (1%) | Loss: 2.071305
Train Epoch: 4 | Batch Status: 1280/50000 (3%) | Loss: 1.771050
Train Epoch: 4 | Batch Status: 1920/50000 (4%) | Loss: 1.540696
Train Epoch: 4 | Batch Status: 2560/50000 (5%) | Loss: 1.846558
Train Epoch: 4 | Batch Status: 3200/50000 (6%) | Loss: 1.874527
Train Epoch: 4 | Batch Status: 3840/50000 (8%) | Loss: 1.742032
Train Epoch: 4 | Batch Status: 4480/50000 (9%) | Loss: 1.533319
Train Epoch: 4 | Batch Status: 5120/50000 (10%) | Loss: 1.629495
Train Epoch: 4 | Batch Status: 5760/50000 (12%) | Loss: 1.567202
Train Epoch: 4 | Batch Status: 6400/50000 (13%) | Loss: 1.579721
Train Epoch: 4 | Batch Status: 7040/50000 (14%) | Loss: 1.719263
Train Epoch: 4 | Batch Status: 7680/50000 (15%) | Loss: 1.715627
Train Epoch: 4 | Batch Status: 8320/50000 (17%) | Loss: 1.779929
Train Epoch: 4 | Batch Status: 8960/50000 (18%) | Loss: 1.544108
Train Epoch: 4 | Batch Status: 9600/50000 (19%) | Loss: 1.678638
Train Epoch: 4 | Batch Status: 10240/50000 (20%) | Loss: 1.328200
Train Epoch: 4 | Batch Status: 10880/50000 (22%) | Loss: 1.668996
Train Epoch: 4 | Batch Status: 11520/50000 (23%) | Loss: 1.658214
Train Epoch: 4 | Batch Status: 12160/50000 (24%) | Loss: 1.418766
Train Epoch: 4 | Batch Status: 12800/50000 (26%) | Loss: 1.639022
Train Epoch: 4 | Batch Status: 13440/50000 (27%) | Loss: 1.680449
Train Epoch: 4 | Batch Status: 14080/50000 (28%) | Loss: 1.650925
Train Epoch: 4 | Batch Status: 14720/50000 (29%) | Loss: 1.521247
Train Epoch: 4 | Batch Status: 15360/50000 (31%) | Loss: 1.577396
Train Epoch: 4 | Batch Status: 16000/50000 (32%) | Loss: 1.839195
Train Epoch: 4 | Batch Status: 16640/50000 (33%) | Loss: 1.699130
Train Epoch: 4 | Batch Status: 17280/50000 (35%) | Loss: 1.740527
Train Epoch: 4 | Batch Status: 17920/50000 (36%) | Loss: 1.600441
Train Epoch: 4 | Batch Status: 18560/50000 (37%) | Loss: 1.579793
Train Epoch: 4 | Batch Status: 19200/50000 (38%) | Loss: 1.558111
Train Epoch: 4 | Batch Status: 19840/50000 (40%) | Loss: 1.541560
Train Epoch: 4 | Batch Status: 20480/50000 (41%) | Loss: 1.637375
Train Epoch: 4 | Batch Status: 21120/50000 (42%) | Loss: 1.836912
Train Epoch: 4 | Batch Status: 21760/50000 (43%) | Loss: 1.670047
Train Epoch: 4 | Batch Status: 22400/50000 (45%) | Loss: 1.567978
Train Epoch: 4 | Batch Status: 23040/50000 (46%) | Loss: 1.688604
Train Epoch: 4 | Batch Status: 23680/50000 (47%) | Loss: 1.756722
Train Epoch: 4 | Batch Status: 24320/50000 (49%) | Loss: 1.659143
Train Epoch: 4 | Batch Status: 24960/50000 (50%) | Loss: 1.627832
Train Epoch: 4 | Batch Status: 25600/50000 (51%) | Loss: 1.779467
Train Epoch: 4 | Batch Status: 26240/50000 (52%) | Loss: 1.615503
Train Epoch: 4 | Batch Status: 26880/50000 (54%) | Loss: 1.894421
Train Epoch: 4 | Batch Status: 27520/50000 (55%) | Loss: 1.575549
Train Epoch: 4 | Batch Status: 28160/50000 (56%) | Loss: 1.533156
Train Epoch: 4 | Batch Status: 28800/50000 (58%) | Loss: 1.447181
Train Epoch: 4 | Batch Status: 29440/50000 (59%) | Loss: 1.590996
Train Epoch: 4 | Batch Status: 30080/50000 (60%) | Loss: 1.575887
Train Epoch: 4 | Batch Status: 30720/50000 (61%) | Loss: 1.758862
Train Epoch: 4 | Batch Status: 31360/50000 (63%) | Loss: 1.614730
Train Epoch: 4 | Batch Status: 32000/50000 (64%) | Loss: 1.582967
Train Epoch: 4 | Batch Status: 32640/50000 (65%) | Loss: 1.758684
Train Epoch: 4 | Batch Status: 33280/50000 (66%) | Loss: 1.391964
Train Epoch: 4 | Batch Status: 33920/50000 (68%) | Loss: 1.545707
Train Epoch: 4 | Batch Status: 34560/50000 (69%) | Loss: 1.709579
Train Epoch: 4 | Batch Status: 35200/50000 (70%) | Loss: 1.756518
Train Epoch: 4 | Batch Status: 35840/50000 (72%) | Loss: 1.515815
Train Epoch: 4 | Batch Status: 36480/50000 (73%) | Loss: 1.388589
Train Epoch: 4 | Batch Status: 37120/50000 (74%) | Loss: 1.800195
Train Epoch: 4 | Batch Status: 37760/50000 (75%) | Loss: 1.411526
Train Epoch: 4 | Batch Status: 38400/50000 (77%) | Loss: 1.489268
Train Epoch: 4 | Batch Status: 39040/50000 (78%) | Loss: 1.648477
Train Epoch: 4 | Batch Status: 39680/50000 (79%) | Loss: 1.580998
Train Epoch: 4 | Batch Status: 40320/50000 (81%) | Loss: 1.678696
Train Epoch: 4 | Batch Status: 40960/50000 (82%) | Loss: 1.615541
Train Epoch: 4 | Batch Status: 41600/50000 (83%) | Loss: 1.506474
Train Epoch: 4 | Batch Status: 42240/50000 (84%) | Loss: 1.494695
Train Epoch: 4 | Batch Status: 42880/50000 (86%) | Loss: 1.610325
Train Epoch: 4 | Batch Status: 43520/50000 (87%) | Loss: 1.355888
Train Epoch: 4 | Batch Status: 44160/50000 (88%) | Loss: 1.495333
Train Epoch: 4 | Batch Status: 44800/50000 (90%) | Loss: 1.539427
Train Epoch: 4 | Batch Status: 45440/50000 (91%) | Loss: 1.665170
Train Epoch: 4 | Batch Status: 46080/50000 (92%) | Loss: 1.554137
Train Epoch: 4 | Batch Status: 46720/50000 (93%) | Loss: 1.705728
Train Epoch: 4 | Batch Status: 47360/50000 (95%) | Loss: 1.552704
Train Epoch: 4 | Batch Status: 48000/50000 (96%) | Loss: 1.616417
Train Epoch: 4 | Batch Status: 48640/50000 (97%) | Loss: 1.596549
Train Epoch: 4 | Batch Status: 49280/50000 (98%) | Loss: 1.505870
Train Epoch: 4 | Batch Status: 49920/50000 (100%) | Loss: 1.864192
Training time: 0m 6s
===========================
Test set: Average loss: 0.0296, Accuracy: 3458/10000 (35%)
Testing time: 0m 7s
Train Epoch: 5 | Batch Status: 0/50000 (0%) | Loss: 1.822173
Train Epoch: 5 | Batch Status: 640/50000 (1%) | Loss: 1.446229
Train Epoch: 5 | Batch Status: 1280/50000 (3%) | Loss: 1.539788
Train Epoch: 5 | Batch Status: 1920/50000 (4%) | Loss: 1.391221
Train Epoch: 5 | Batch Status: 2560/50000 (5%) | Loss: 1.592271
Train Epoch: 5 | Batch Status: 3200/50000 (6%) | Loss: 1.547070
Train Epoch: 5 | Batch Status: 3840/50000 (8%) | Loss: 1.304626
Train Epoch: 5 | Batch Status: 4480/50000 (9%) | Loss: 1.373312
Train Epoch: 5 | Batch Status: 5120/50000 (10%) | Loss: 1.358480
Train Epoch: 5 | Batch Status: 5760/50000 (12%) | Loss: 1.583030
Train Epoch: 5 | Batch Status: 6400/50000 (13%) | Loss: 1.904695
Train Epoch: 5 | Batch Status: 7040/50000 (14%) | Loss: 1.543920
Train Epoch: 5 | Batch Status: 7680/50000 (15%) | Loss: 1.424248
Train Epoch: 5 | Batch Status: 8320/50000 (17%) | Loss: 1.502147
Train Epoch: 5 | Batch Status: 8960/50000 (18%) | Loss: 1.715775
Train Epoch: 5 | Batch Status: 9600/50000 (19%) | Loss: 1.619118
Train Epoch: 5 | Batch Status: 10240/50000 (20%) | Loss: 1.470978
Train Epoch: 5 | Batch Status: 10880/50000 (22%) | Loss: 1.705833
Train Epoch: 5 | Batch Status: 11520/50000 (23%) | Loss: 1.537857
Train Epoch: 5 | Batch Status: 12160/50000 (24%) | Loss: 1.671722
Train Epoch: 5 | Batch Status: 12800/50000 (26%) | Loss: 1.515812
Train Epoch: 5 | Batch Status: 13440/50000 (27%) | Loss: 1.623387
Train Epoch: 5 | Batch Status: 14080/50000 (28%) | Loss: 1.703303
Train Epoch: 5 | Batch Status: 14720/50000 (29%) | Loss: 1.579847
Train Epoch: 5 | Batch Status: 15360/50000 (31%) | Loss: 1.448100
Train Epoch: 5 | Batch Status: 16000/50000 (32%) | Loss: 1.436941
Train Epoch: 5 | Batch Status: 16640/50000 (33%) | Loss: 1.421772
Train Epoch: 5 | Batch Status: 17280/50000 (35%) | Loss: 1.530502
Train Epoch: 5 | Batch Status: 17920/50000 (36%) | Loss: 1.189285
Train Epoch: 5 | Batch Status: 18560/50000 (37%) | Loss: 1.227402
Train Epoch: 5 | Batch Status: 19200/50000 (38%) | Loss: 1.457553
Train Epoch: 5 | Batch Status: 19840/50000 (40%) | Loss: 1.659267
Train Epoch: 5 | Batch Status: 20480/50000 (41%) | Loss: 1.468486
Train Epoch: 5 | Batch Status: 21120/50000 (42%) | Loss: 1.721519
Train Epoch: 5 | Batch Status: 21760/50000 (43%) | Loss: 1.567099
Train Epoch: 5 | Batch Status: 22400/50000 (45%) | Loss: 1.549762
Train Epoch: 5 | Batch Status: 23040/50000 (46%) | Loss: 1.579604
Train Epoch: 5 | Batch Status: 23680/50000 (47%) | Loss: 1.566174
Train Epoch: 5 | Batch Status: 24320/50000 (49%) | Loss: 1.415713
Train Epoch: 5 | Batch Status: 24960/50000 (50%) | Loss: 1.686714
Train Epoch: 5 | Batch Status: 25600/50000 (51%) | Loss: 1.503157
Train Epoch: 5 | Batch Status: 26240/50000 (52%) | Loss: 1.476280
Train Epoch: 5 | Batch Status: 26880/50000 (54%) | Loss: 1.484453
Train Epoch: 5 | Batch Status: 27520/50000 (55%) | Loss: 1.371297
Train Epoch: 5 | Batch Status: 28160/50000 (56%) | Loss: 1.497889
Train Epoch: 5 | Batch Status: 28800/50000 (58%) | Loss: 1.432095
Train Epoch: 5 | Batch Status: 29440/50000 (59%) | Loss: 1.651112
Train Epoch: 5 | Batch Status: 30080/50000 (60%) | Loss: 1.381632
Train Epoch: 5 | Batch Status: 30720/50000 (61%) | Loss: 1.997741
Train Epoch: 5 | Batch Status: 31360/50000 (63%) | Loss: 1.462149
Train Epoch: 5 | Batch Status: 32000/50000 (64%) | Loss: 1.709768
Train Epoch: 5 | Batch Status: 32640/50000 (65%) | Loss: 1.332987
Train Epoch: 5 | Batch Status: 33280/50000 (66%) | Loss: 1.524412
Train Epoch: 5 | Batch Status: 33920/50000 (68%) | Loss: 1.307634
Train Epoch: 5 | Batch Status: 34560/50000 (69%) | Loss: 1.513311
Train Epoch: 5 | Batch Status: 35200/50000 (70%) | Loss: 1.434103
Train Epoch: 5 | Batch Status: 35840/50000 (72%) | Loss: 1.359136
Train Epoch: 5 | Batch Status: 36480/50000 (73%) | Loss: 1.516614
Train Epoch: 5 | Batch Status: 37120/50000 (74%) | Loss: 1.720983
Train Epoch: 5 | Batch Status: 37760/50000 (75%) | Loss: 1.435341
Train Epoch: 5 | Batch Status: 38400/50000 (77%) | Loss: 1.598479
Train Epoch: 5 | Batch Status: 39040/50000 (78%) | Loss: 1.453193
Train Epoch: 5 | Batch Status: 39680/50000 (79%) | Loss: 1.622408
Train Epoch: 5 | Batch Status: 40320/50000 (81%) | Loss: 1.612339
Train Epoch: 5 | Batch Status: 40960/50000 (82%) | Loss: 1.420392
Train Epoch: 5 | Batch Status: 41600/50000 (83%) | Loss: 1.476946
Train Epoch: 5 | Batch Status: 42240/50000 (84%) | Loss: 1.533992
Train Epoch: 5 | Batch Status: 42880/50000 (86%) | Loss: 1.496833
Train Epoch: 5 | Batch Status: 43520/50000 (87%) | Loss: 1.453892
Train Epoch: 5 | Batch Status: 44160/50000 (88%) | Loss: 1.578845
Train Epoch: 5 | Batch Status: 44800/50000 (90%) | Loss: 1.644650
Train Epoch: 5 | Batch Status: 45440/50000 (91%) | Loss: 1.498288
Train Epoch: 5 | Batch Status: 46080/50000 (92%) | Loss: 1.779317
Train Epoch: 5 | Batch Status: 46720/50000 (93%) | Loss: 1.365582
Train Epoch: 5 | Batch Status: 47360/50000 (95%) | Loss: 1.485359
Train Epoch: 5 | Batch Status: 48000/50000 (96%) | Loss: 1.393395
Train Epoch: 5 | Batch Status: 48640/50000 (97%) | Loss: 1.379532
Train Epoch: 5 | Batch Status: 49280/50000 (98%) | Loss: 1.481343
Train Epoch: 5 | Batch Status: 49920/50000 (100%) | Loss: 1.451552
Training time: 0m 6s
===========================
Test set: Average loss: 0.0249, Accuracy: 4324/10000 (43%)
Testing time: 0m 7s
Train Epoch: 6 | Batch Status: 0/50000 (0%) | Loss: 1.383267
Train Epoch: 6 | Batch Status: 640/50000 (1%) | Loss: 1.461397
Train Epoch: 6 | Batch Status: 1280/50000 (3%) | Loss: 1.544568
Train Epoch: 6 | Batch Status: 1920/50000 (4%) | Loss: 1.471427
Train Epoch: 6 | Batch Status: 2560/50000 (5%) | Loss: 1.314599
Train Epoch: 6 | Batch Status: 3200/50000 (6%) | Loss: 1.213807
Train Epoch: 6 | Batch Status: 3840/50000 (8%) | Loss: 1.751316
Train Epoch: 6 | Batch Status: 4480/50000 (9%) | Loss: 1.481106
Train Epoch: 6 | Batch Status: 5120/50000 (10%) | Loss: 1.719940
Train Epoch: 6 | Batch Status: 5760/50000 (12%) | Loss: 1.594771
Train Epoch: 6 | Batch Status: 6400/50000 (13%) | Loss: 1.412328
Train Epoch: 6 | Batch Status: 7040/50000 (14%) | Loss: 1.372475
Train Epoch: 6 | Batch Status: 7680/50000 (15%) | Loss: 1.498523
Train Epoch: 6 | Batch Status: 8320/50000 (17%) | Loss: 1.471356
Train Epoch: 6 | Batch Status: 8960/50000 (18%) | Loss: 1.490124
Train Epoch: 6 | Batch Status: 9600/50000 (19%) | Loss: 1.314925
Train Epoch: 6 | Batch Status: 10240/50000 (20%) | Loss: 1.536335
Train Epoch: 6 | Batch Status: 10880/50000 (22%) | Loss: 1.514332
Train Epoch: 6 | Batch Status: 11520/50000 (23%) | Loss: 1.552904
Train Epoch: 6 | Batch Status: 12160/50000 (24%) | Loss: 1.515995
Train Epoch: 6 | Batch Status: 12800/50000 (26%) | Loss: 1.691048
Train Epoch: 6 | Batch Status: 13440/50000 (27%) | Loss: 1.494314
Train Epoch: 6 | Batch Status: 14080/50000 (28%) | Loss: 1.437325
Train Epoch: 6 | Batch Status: 14720/50000 (29%) | Loss: 1.332118
Train Epoch: 6 | Batch Status: 15360/50000 (31%) | Loss: 1.366856
Train Epoch: 6 | Batch Status: 16000/50000 (32%) | Loss: 1.592311
Train Epoch: 6 | Batch Status: 16640/50000 (33%) | Loss: 1.673827
Train Epoch: 6 | Batch Status: 17280/50000 (35%) | Loss: 1.352833
Train Epoch: 6 | Batch Status: 17920/50000 (36%) | Loss: 1.567802
Train Epoch: 6 | Batch Status: 18560/50000 (37%) | Loss: 1.345707
Train Epoch: 6 | Batch Status: 19200/50000 (38%) | Loss: 1.441437
Train Epoch: 6 | Batch Status: 19840/50000 (40%) | Loss: 1.612059
Train Epoch: 6 | Batch Status: 20480/50000 (41%) | Loss: 1.473103
Train Epoch: 6 | Batch Status: 21120/50000 (42%) | Loss: 1.494641
Train Epoch: 6 | Batch Status: 21760/50000 (43%) | Loss: 1.509014
Train Epoch: 6 | Batch Status: 22400/50000 (45%) | Loss: 1.338446
Train Epoch: 6 | Batch Status: 23040/50000 (46%) | Loss: 1.378332
Train Epoch: 6 | Batch Status: 23680/50000 (47%) | Loss: 1.318803
Train Epoch: 6 | Batch Status: 24320/50000 (49%) | Loss: 1.653300
Train Epoch: 6 | Batch Status: 24960/50000 (50%) | Loss: 1.519131
Train Epoch: 6 | Batch Status: 25600/50000 (51%) | Loss: 1.439601
Train Epoch: 6 | Batch Status: 26240/50000 (52%) | Loss: 1.457190
Train Epoch: 6 | Batch Status: 26880/50000 (54%) | Loss: 1.435826
Train Epoch: 6 | Batch Status: 27520/50000 (55%) | Loss: 1.296788
Train Epoch: 6 | Batch Status: 28160/50000 (56%) | Loss: 1.329495
Train Epoch: 6 | Batch Status: 28800/50000 (58%) | Loss: 1.502229
Train Epoch: 6 | Batch Status: 29440/50000 (59%) | Loss: 1.383356
Train Epoch: 6 | Batch Status: 30080/50000 (60%) | Loss: 1.610050
Train Epoch: 6 | Batch Status: 30720/50000 (61%) | Loss: 1.262730
Train Epoch: 6 | Batch Status: 31360/50000 (63%) | Loss: 1.536590
Train Epoch: 6 | Batch Status: 32000/50000 (64%) | Loss: 1.631058
Train Epoch: 6 | Batch Status: 32640/50000 (65%) | Loss: 1.476357
Train Epoch: 6 | Batch Status: 33280/50000 (66%) | Loss: 1.354637
Train Epoch: 6 | Batch Status: 33920/50000 (68%) | Loss: 1.688624
Train Epoch: 6 | Batch Status: 34560/50000 (69%) | Loss: 1.539448
Train Epoch: 6 | Batch Status: 35200/50000 (70%) | Loss: 1.513189
Train Epoch: 6 | Batch Status: 35840/50000 (72%) | Loss: 1.718433
Train Epoch: 6 | Batch Status: 36480/50000 (73%) | Loss: 1.551477
Train Epoch: 6 | Batch Status: 37120/50000 (74%) | Loss: 1.405339
Train Epoch: 6 | Batch Status: 37760/50000 (75%) | Loss: 1.423458
Train Epoch: 6 | Batch Status: 38400/50000 (77%) | Loss: 1.600654
Train Epoch: 6 | Batch Status: 39040/50000 (78%) | Loss: 1.435440
Train Epoch: 6 | Batch Status: 39680/50000 (79%) | Loss: 1.465233
Train Epoch: 6 | Batch Status: 40320/50000 (81%) | Loss: 1.281727
Train Epoch: 6 | Batch Status: 40960/50000 (82%) | Loss: 1.366308
Train Epoch: 6 | Batch Status: 41600/50000 (83%) | Loss: 1.474794
Train Epoch: 6 | Batch Status: 42240/50000 (84%) | Loss: 1.541199
Train Epoch: 6 | Batch Status: 42880/50000 (86%) | Loss: 1.362571
Train Epoch: 6 | Batch Status: 43520/50000 (87%) | Loss: 1.360917
Train Epoch: 6 | Batch Status: 44160/50000 (88%) | Loss: 1.415277
Train Epoch: 6 | Batch Status: 44800/50000 (90%) | Loss: 1.525495
Train Epoch: 6 | Batch Status: 45440/50000 (91%) | Loss: 1.578268
Train Epoch: 6 | Batch Status: 46080/50000 (92%) | Loss: 1.423885
Train Epoch: 6 | Batch Status: 46720/50000 (93%) | Loss: 1.528280
Train Epoch: 6 | Batch Status: 47360/50000 (95%) | Loss: 1.542479
Train Epoch: 6 | Batch Status: 48000/50000 (96%) | Loss: 1.411813
Train Epoch: 6 | Batch Status: 48640/50000 (97%) | Loss: 1.512930
Train Epoch: 6 | Batch Status: 49280/50000 (98%) | Loss: 1.643258
Train Epoch: 6 | Batch Status: 49920/50000 (100%) | Loss: 1.560114
Training time: 0m 6s
===========================
Test set: Average loss: 0.0257, Accuracy: 4258/10000 (43%)
Testing time: 0m 7s
Train Epoch: 7 | Batch Status: 0/50000 (0%) | Loss: 1.303095
Train Epoch: 7 | Batch Status: 640/50000 (1%) | Loss: 1.396807
Train Epoch: 7 | Batch Status: 1280/50000 (3%) | Loss: 1.390447
Train Epoch: 7 | Batch Status: 1920/50000 (4%) | Loss: 1.700904
Train Epoch: 7 | Batch Status: 2560/50000 (5%) | Loss: 1.347698
Train Epoch: 7 | Batch Status: 3200/50000 (6%) | Loss: 1.456709
Train Epoch: 7 | Batch Status: 3840/50000 (8%) | Loss: 1.274194
Train Epoch: 7 | Batch Status: 4480/50000 (9%) | Loss: 1.359151
Train Epoch: 7 | Batch Status: 5120/50000 (10%) | Loss: 1.368525
Train Epoch: 7 | Batch Status: 5760/50000 (12%) | Loss: 1.520443
Train Epoch: 7 | Batch Status: 6400/50000 (13%) | Loss: 1.712659
Train Epoch: 7 | Batch Status: 7040/50000 (14%) | Loss: 1.296550
Train Epoch: 7 | Batch Status: 7680/50000 (15%) | Loss: 1.637812
Train Epoch: 7 | Batch Status: 8320/50000 (17%) | Loss: 1.372432
Train Epoch: 7 | Batch Status: 8960/50000 (18%) | Loss: 1.730750
Train Epoch: 7 | Batch Status: 9600/50000 (19%) | Loss: 1.246298
Train Epoch: 7 | Batch Status: 10240/50000 (20%) | Loss: 1.468355
Train Epoch: 7 | Batch Status: 10880/50000 (22%) | Loss: 1.441489
Train Epoch: 7 | Batch Status: 11520/50000 (23%) | Loss: 1.525300
Train Epoch: 7 | Batch Status: 12160/50000 (24%) | Loss: 1.476217
Train Epoch: 7 | Batch Status: 12800/50000 (26%) | Loss: 1.410931
Train Epoch: 7 | Batch Status: 13440/50000 (27%) | Loss: 1.364926
Train Epoch: 7 | Batch Status: 14080/50000 (28%) | Loss: 1.515189
Train Epoch: 7 | Batch Status: 14720/50000 (29%) | Loss: 1.232523
Train Epoch: 7 | Batch Status: 15360/50000 (31%) | Loss: 1.690650
Train Epoch: 7 | Batch Status: 16000/50000 (32%) | Loss: 1.660186
Train Epoch: 7 | Batch Status: 16640/50000 (33%) | Loss: 1.404513
Train Epoch: 7 | Batch Status: 17280/50000 (35%) | Loss: 1.431359
Train Epoch: 7 | Batch Status: 17920/50000 (36%) | Loss: 1.292485
Train Epoch: 7 | Batch Status: 18560/50000 (37%) | Loss: 1.328373
Train Epoch: 7 | Batch Status: 19200/50000 (38%) | Loss: 1.642239
Train Epoch: 7 | Batch Status: 19840/50000 (40%) | Loss: 1.542332
Train Epoch: 7 | Batch Status: 20480/50000 (41%) | Loss: 1.504606
Train Epoch: 7 | Batch Status: 21120/50000 (42%) | Loss: 1.443584
Train Epoch: 7 | Batch Status: 21760/50000 (43%) | Loss: 1.429955
Train Epoch: 7 | Batch Status: 22400/50000 (45%) | Loss: 1.636142
Train Epoch: 7 | Batch Status: 23040/50000 (46%) | Loss: 1.606086
Train Epoch: 7 | Batch Status: 23680/50000 (47%) | Loss: 1.265074
Train Epoch: 7 | Batch Status: 24320/50000 (49%) | Loss: 1.260653
Train Epoch: 7 | Batch Status: 24960/50000 (50%) | Loss: 1.508955
Train Epoch: 7 | Batch Status: 25600/50000 (51%) | Loss: 1.565314
Train Epoch: 7 | Batch Status: 26240/50000 (52%) | Loss: 1.101512
Train Epoch: 7 | Batch Status: 26880/50000 (54%) | Loss: 1.511820
Train Epoch: 7 | Batch Status: 27520/50000 (55%) | Loss: 1.414820
Train Epoch: 7 | Batch Status: 28160/50000 (56%) | Loss: 1.283070
Train Epoch: 7 | Batch Status: 28800/50000 (58%) | Loss: 1.421770
Train Epoch: 7 | Batch Status: 29440/50000 (59%) | Loss: 1.388011
Train Epoch: 7 | Batch Status: 30080/50000 (60%) | Loss: 1.395599
Train Epoch: 7 | Batch Status: 30720/50000 (61%) | Loss: 1.349111
Train Epoch: 7 | Batch Status: 31360/50000 (63%) | Loss: 1.489599
Train Epoch: 7 | Batch Status: 32000/50000 (64%) | Loss: 1.421054
Train Epoch: 7 | Batch Status: 32640/50000 (65%) | Loss: 1.375566
Train Epoch: 7 | Batch Status: 33280/50000 (66%) | Loss: 1.542711
Train Epoch: 7 | Batch Status: 33920/50000 (68%) | Loss: 1.477750
Train Epoch: 7 | Batch Status: 34560/50000 (69%) | Loss: 1.434626
Train Epoch: 7 | Batch Status: 35200/50000 (70%) | Loss: 1.355645
Train Epoch: 7 | Batch Status: 35840/50000 (72%) | Loss: 1.166381
Train Epoch: 7 | Batch Status: 36480/50000 (73%) | Loss: 1.448404
Train Epoch: 7 | Batch Status: 37120/50000 (74%) | Loss: 1.408027
Train Epoch: 7 | Batch Status: 37760/50000 (75%) | Loss: 1.406300
Train Epoch: 7 | Batch Status: 38400/50000 (77%) | Loss: 1.543408
Train Epoch: 7 | Batch Status: 39040/50000 (78%) | Loss: 1.410562
Train Epoch: 7 | Batch Status: 39680/50000 (79%) | Loss: 1.362337
Train Epoch: 7 | Batch Status: 40320/50000 (81%) | Loss: 1.356714
Train Epoch: 7 | Batch Status: 40960/50000 (82%) | Loss: 1.455901
Train Epoch: 7 | Batch Status: 41600/50000 (83%) | Loss: 1.465639
Train Epoch: 7 | Batch Status: 42240/50000 (84%) | Loss: 1.382976
Train Epoch: 7 | Batch Status: 42880/50000 (86%) | Loss: 1.492184
Train Epoch: 7 | Batch Status: 43520/50000 (87%) | Loss: 1.491296
Train Epoch: 7 | Batch Status: 44160/50000 (88%) | Loss: 1.434966
Train Epoch: 7 | Batch Status: 44800/50000 (90%) | Loss: 1.331445
Train Epoch: 7 | Batch Status: 45440/50000 (91%) | Loss: 1.433333
Train Epoch: 7 | Batch Status: 46080/50000 (92%) | Loss: 1.522799
Train Epoch: 7 | Batch Status: 46720/50000 (93%) | Loss: 1.372979
Train Epoch: 7 | Batch Status: 47360/50000 (95%) | Loss: 1.356832
Train Epoch: 7 | Batch Status: 48000/50000 (96%) | Loss: 1.532097
Train Epoch: 7 | Batch Status: 48640/50000 (97%) | Loss: 1.232276
Train Epoch: 7 | Batch Status: 49280/50000 (98%) | Loss: 1.347396
Train Epoch: 7 | Batch Status: 49920/50000 (100%) | Loss: 1.459254
Training time: 0m 6s
===========================
Test set: Average loss: 0.0256, Accuracy: 4053/10000 (41%)
Testing time: 0m 7s
Train Epoch: 8 | Batch Status: 0/50000 (0%) | Loss: 1.754014
Train Epoch: 8 | Batch Status: 640/50000 (1%) | Loss: 1.198841
Train Epoch: 8 | Batch Status: 1280/50000 (3%) | Loss: 1.299282
Train Epoch: 8 | Batch Status: 1920/50000 (4%) | Loss: 1.479539
Train Epoch: 8 | Batch Status: 2560/50000 (5%) | Loss: 1.339682
Train Epoch: 8 | Batch Status: 3200/50000 (6%) | Loss: 1.168061
Train Epoch: 8 | Batch Status: 3840/50000 (8%) | Loss: 1.409980
Train Epoch: 8 | Batch Status: 4480/50000 (9%) | Loss: 1.461143
Train Epoch: 8 | Batch Status: 5120/50000 (10%) | Loss: 1.077085
Train Epoch: 8 | Batch Status: 5760/50000 (12%) | Loss: 1.304958
Train Epoch: 8 | Batch Status: 6400/50000 (13%) | Loss: 1.496707
Train Epoch: 8 | Batch Status: 7040/50000 (14%) | Loss: 1.314318
Train Epoch: 8 | Batch Status: 7680/50000 (15%) | Loss: 1.283203
Train Epoch: 8 | Batch Status: 8320/50000 (17%) | Loss: 1.368049
Train Epoch: 8 | Batch Status: 8960/50000 (18%) | Loss: 1.334846
Train Epoch: 8 | Batch Status: 9600/50000 (19%) | Loss: 1.573992
Train Epoch: 8 | Batch Status: 10240/50000 (20%) | Loss: 1.567191
Train Epoch: 8 | Batch Status: 10880/50000 (22%) | Loss: 1.403157
Train Epoch: 8 | Batch Status: 11520/50000 (23%) | Loss: 1.420794
Train Epoch: 8 | Batch Status: 12160/50000 (24%) | Loss: 1.247713
Train Epoch: 8 | Batch Status: 12800/50000 (26%) | Loss: 1.606192
Train Epoch: 8 | Batch Status: 13440/50000 (27%) | Loss: 1.401894
Train Epoch: 8 | Batch Status: 14080/50000 (28%) | Loss: 1.267167
Train Epoch: 8 | Batch Status: 14720/50000 (29%) | Loss: 1.277706
Train Epoch: 8 | Batch Status: 15360/50000 (31%) | Loss: 1.216243
Train Epoch: 8 | Batch Status: 16000/50000 (32%) | Loss: 1.727769
Train Epoch: 8 | Batch Status: 16640/50000 (33%) | Loss: 1.584745
Train Epoch: 8 | Batch Status: 17280/50000 (35%) | Loss: 1.499296
Train Epoch: 8 | Batch Status: 17920/50000 (36%) | Loss: 1.528489
Train Epoch: 8 | Batch Status: 18560/50000 (37%) | Loss: 1.605172
Train Epoch: 8 | Batch Status: 19200/50000 (38%) | Loss: 1.584331
Train Epoch: 8 | Batch Status: 19840/50000 (40%) | Loss: 1.405048
Train Epoch: 8 | Batch Status: 20480/50000 (41%) | Loss: 1.517398
Train Epoch: 8 | Batch Status: 21120/50000 (42%) | Loss: 1.316639
Train Epoch: 8 | Batch Status: 21760/50000 (43%) | Loss: 1.547871
Train Epoch: 8 | Batch Status: 22400/50000 (45%) | Loss: 1.366493
Train Epoch: 8 | Batch Status: 23040/50000 (46%) | Loss: 1.362014
Train Epoch: 8 | Batch Status: 23680/50000 (47%) | Loss: 1.327509
Train Epoch: 8 | Batch Status: 24320/50000 (49%) | Loss: 1.517857
Train Epoch: 8 | Batch Status: 24960/50000 (50%) | Loss: 1.552727
Train Epoch: 8 | Batch Status: 25600/50000 (51%) | Loss: 1.488614
Train Epoch: 8 | Batch Status: 26240/50000 (52%) | Loss: 1.322248
Train Epoch: 8 | Batch Status: 26880/50000 (54%) | Loss: 1.284499
Train Epoch: 8 | Batch Status: 27520/50000 (55%) | Loss: 1.396475
Train Epoch: 8 | Batch Status: 28160/50000 (56%) | Loss: 1.432695
Train Epoch: 8 | Batch Status: 28800/50000 (58%) | Loss: 1.351068
Train Epoch: 8 | Batch Status: 29440/50000 (59%) | Loss: 1.256220
Train Epoch: 8 | Batch Status: 30080/50000 (60%) | Loss: 1.538403
Train Epoch: 8 | Batch Status: 30720/50000 (61%) | Loss: 1.538070
Train Epoch: 8 | Batch Status: 31360/50000 (63%) | Loss: 1.194399
Train Epoch: 8 | Batch Status: 32000/50000 (64%) | Loss: 1.289367
Train Epoch: 8 | Batch Status: 32640/50000 (65%) | Loss: 1.183278
Train Epoch: 8 | Batch Status: 33280/50000 (66%) | Loss: 1.599021
Train Epoch: 8 | Batch Status: 33920/50000 (68%) | Loss: 1.309098
Train Epoch: 8 | Batch Status: 34560/50000 (69%) | Loss: 1.488415
Train Epoch: 8 | Batch Status: 35200/50000 (70%) | Loss: 1.476779
Train Epoch: 8 | Batch Status: 35840/50000 (72%) | Loss: 1.509902
Train Epoch: 8 | Batch Status: 36480/50000 (73%) | Loss: 1.421737
Train Epoch: 8 | Batch Status: 37120/50000 (74%) | Loss: 1.316038
Train Epoch: 8 | Batch Status: 37760/50000 (75%) | Loss: 1.198761
Train Epoch: 8 | Batch Status: 38400/50000 (77%) | Loss: 1.368834
Train Epoch: 8 | Batch Status: 39040/50000 (78%) | Loss: 1.263400
Train Epoch: 8 | Batch Status: 39680/50000 (79%) | Loss: 1.415114
Train Epoch: 8 | Batch Status: 40320/50000 (81%) | Loss: 1.034247
Train Epoch: 8 | Batch Status: 40960/50000 (82%) | Loss: 1.411877
Train Epoch: 8 | Batch Status: 41600/50000 (83%) | Loss: 1.678122
Train Epoch: 8 | Batch Status: 42240/50000 (84%) | Loss: 1.398896
Train Epoch: 8 | Batch Status: 42880/50000 (86%) | Loss: 1.630100
Train Epoch: 8 | Batch Status: 43520/50000 (87%) | Loss: 0.986156
Train Epoch: 8 | Batch Status: 44160/50000 (88%) | Loss: 1.209832
Train Epoch: 8 | Batch Status: 44800/50000 (90%) | Loss: 1.509290
Train Epoch: 8 | Batch Status: 45440/50000 (91%) | Loss: 1.195759
Train Epoch: 8 | Batch Status: 46080/50000 (92%) | Loss: 1.328167
Train Epoch: 8 | Batch Status: 46720/50000 (93%) | Loss: 1.322575
Train Epoch: 8 | Batch Status: 47360/50000 (95%) | Loss: 1.395520
Train Epoch: 8 | Batch Status: 48000/50000 (96%) | Loss: 1.482855
Train Epoch: 8 | Batch Status: 48640/50000 (97%) | Loss: 1.340259
Train Epoch: 8 | Batch Status: 49280/50000 (98%) | Loss: 1.564031
Train Epoch: 8 | Batch Status: 49920/50000 (100%) | Loss: 1.542186
Training time: 0m 6s
===========================
Test set: Average loss: 0.0239, Accuracy: 4600/10000 (46%)
Testing time: 0m 7s
Train Epoch: 9 | Batch Status: 0/50000 (0%) | Loss: 1.451973
Train Epoch: 9 | Batch Status: 640/50000 (1%) | Loss: 1.372529
Train Epoch: 9 | Batch Status: 1280/50000 (3%) | Loss: 1.491306
Train Epoch: 9 | Batch Status: 1920/50000 (4%) | Loss: 1.311235
Train Epoch: 9 | Batch Status: 2560/50000 (5%) | Loss: 1.212537
Train Epoch: 9 | Batch Status: 3200/50000 (6%) | Loss: 1.257376
Train Epoch: 9 | Batch Status: 3840/50000 (8%) | Loss: 1.262114
Train Epoch: 9 | Batch Status: 4480/50000 (9%) | Loss: 1.608002
Train Epoch: 9 | Batch Status: 5120/50000 (10%) | Loss: 1.277098
Train Epoch: 9 | Batch Status: 5760/50000 (12%) | Loss: 1.346457
Train Epoch: 9 | Batch Status: 6400/50000 (13%) | Loss: 1.333298
Train Epoch: 9 | Batch Status: 7040/50000 (14%) | Loss: 1.118773
Train Epoch: 9 | Batch Status: 7680/50000 (15%) | Loss: 1.463365
Train Epoch: 9 | Batch Status: 8320/50000 (17%) | Loss: 1.306254
Train Epoch: 9 | Batch Status: 8960/50000 (18%) | Loss: 1.414467
Train Epoch: 9 | Batch Status: 9600/50000 (19%) | Loss: 1.170205
Train Epoch: 9 | Batch Status: 10240/50000 (20%) | Loss: 1.328419
Train Epoch: 9 | Batch Status: 10880/50000 (22%) | Loss: 1.590486
Train Epoch: 9 | Batch Status: 11520/50000 (23%) | Loss: 1.026766
Train Epoch: 9 | Batch Status: 12160/50000 (24%) | Loss: 1.142545
Train Epoch: 9 | Batch Status: 12800/50000 (26%) | Loss: 1.307917
Train Epoch: 9 | Batch Status: 13440/50000 (27%) | Loss: 1.403128
Train Epoch: 9 | Batch Status: 14080/50000 (28%) | Loss: 1.272921
Train Epoch: 9 | Batch Status: 14720/50000 (29%) | Loss: 1.538980
Train Epoch: 9 | Batch Status: 15360/50000 (31%) | Loss: 1.456506
Train Epoch: 9 | Batch Status: 16000/50000 (32%) | Loss: 1.072250
Train Epoch: 9 | Batch Status: 16640/50000 (33%) | Loss: 1.194555
Train Epoch: 9 | Batch Status: 17280/50000 (35%) | Loss: 1.484896
Train Epoch: 9 | Batch Status: 17920/50000 (36%) | Loss: 1.203474
Train Epoch: 9 | Batch Status: 18560/50000 (37%) | Loss: 1.592532
Train Epoch: 9 | Batch Status: 19200/50000 (38%) | Loss: 1.201919
Train Epoch: 9 | Batch Status: 19840/50000 (40%) | Loss: 1.259140
Train Epoch: 9 | Batch Status: 20480/50000 (41%) | Loss: 1.463423
Train Epoch: 9 | Batch Status: 21120/50000 (42%) | Loss: 1.184092
Train Epoch: 9 | Batch Status: 21760/50000 (43%) | Loss: 1.131511
Train Epoch: 9 | Batch Status: 22400/50000 (45%) | Loss: 1.118655
Train Epoch: 9 | Batch Status: 23040/50000 (46%) | Loss: 1.386844
Train Epoch: 9 | Batch Status: 23680/50000 (47%) | Loss: 1.463819
Train Epoch: 9 | Batch Status: 24320/50000 (49%) | Loss: 1.397107
Train Epoch: 9 | Batch Status: 24960/50000 (50%) | Loss: 1.223579
Train Epoch: 9 | Batch Status: 25600/50000 (51%) | Loss: 1.431403
Train Epoch: 9 | Batch Status: 26240/50000 (52%) | Loss: 1.462393
Train Epoch: 9 | Batch Status: 26880/50000 (54%) | Loss: 1.320816
Train Epoch: 9 | Batch Status: 27520/50000 (55%) | Loss: 1.213859
Train Epoch: 9 | Batch Status: 28160/50000 (56%) | Loss: 1.201841
Train Epoch: 9 | Batch Status: 28800/50000 (58%) | Loss: 1.184067
Train Epoch: 9 | Batch Status: 29440/50000 (59%) | Loss: 1.503456
Train Epoch: 9 | Batch Status: 30080/50000 (60%) | Loss: 1.266485
Train Epoch: 9 | Batch Status: 30720/50000 (61%) | Loss: 1.324017
Train Epoch: 9 | Batch Status: 31360/50000 (63%) | Loss: 1.410979
Train Epoch: 9 | Batch Status: 32000/50000 (64%) | Loss: 1.328564
Train Epoch: 9 | Batch Status: 32640/50000 (65%) | Loss: 1.578736
Train Epoch: 9 | Batch Status: 33280/50000 (66%) | Loss: 1.375774
Train Epoch: 9 | Batch Status: 33920/50000 (68%) | Loss: 1.267152
Train Epoch: 9 | Batch Status: 34560/50000 (69%) | Loss: 1.285454
Train Epoch: 9 | Batch Status: 35200/50000 (70%) | Loss: 1.503320
Train Epoch: 9 | Batch Status: 35840/50000 (72%) | Loss: 1.426221
Train Epoch: 9 | Batch Status: 36480/50000 (73%) | Loss: 1.242847
Train Epoch: 9 | Batch Status: 37120/50000 (74%) | Loss: 1.546647
Train Epoch: 9 | Batch Status: 37760/50000 (75%) | Loss: 1.297792
Train Epoch: 9 | Batch Status: 38400/50000 (77%) | Loss: 1.232499
Train Epoch: 9 | Batch Status: 39040/50000 (78%) | Loss: 1.364900
Train Epoch: 9 | Batch Status: 39680/50000 (79%) | Loss: 1.101284
Train Epoch: 9 | Batch Status: 40320/50000 (81%) | Loss: 1.446041
Train Epoch: 9 | Batch Status: 40960/50000 (82%) | Loss: 1.276941
Train Epoch: 9 | Batch Status: 41600/50000 (83%) | Loss: 1.299430
Train Epoch: 9 | Batch Status: 42240/50000 (84%) | Loss: 1.217729
Train Epoch: 9 | Batch Status: 42880/50000 (86%) | Loss: 1.502466
Train Epoch: 9 | Batch Status: 43520/50000 (87%) | Loss: 1.208186
Train Epoch: 9 | Batch Status: 44160/50000 (88%) | Loss: 1.407407
Train Epoch: 9 | Batch Status: 44800/50000 (90%) | Loss: 1.217129
Train Epoch: 9 | Batch Status: 45440/50000 (91%) | Loss: 1.448588
Train Epoch: 9 | Batch Status: 46080/50000 (92%) | Loss: 1.299772
Train Epoch: 9 | Batch Status: 46720/50000 (93%) | Loss: 1.173299
Train Epoch: 9 | Batch Status: 47360/50000 (95%) | Loss: 1.167305
Train Epoch: 9 | Batch Status: 48000/50000 (96%) | Loss: 1.207888
Train Epoch: 9 | Batch Status: 48640/50000 (97%) | Loss: 1.405597
Train Epoch: 9 | Batch Status: 49280/50000 (98%) | Loss: 1.237440
Train Epoch: 9 | Batch Status: 49920/50000 (100%) | Loss: 1.252822
Training time: 0m 6s
===========================
Test set: Average loss: 0.0224, Accuracy: 4977/10000 (50%)
Testing time: 0m 7s
Total Time: 1m 2s
Model was trained on cuda
