from __future__ import print_function # __future__에서 print_function을 불러옴
from torch import nn, optim, cuda # torch에서 nn, optim, cuda를 불러옴
from torch.utils import data  # torch.utils에서 data를 불러옴
from torchvision import datasets, transforms as transforms  # torchvision에서 datasets, transforms를 불러옴
import torch
import torchvision
import torch.nn.functional as F # torch.nn.functional을 F라 선언
import time

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']    # CIFAR10 데이터 의 10가지 클래스

# Training settings
batch_size = 64

device = 'cuda' if cuda.is_available() else 'cpu' # 가능하다면 'cuda'(GPU)가속을 사용
print(f'Training CIFAR10 Model on {device}\n{"=" * 44}')

# CIFAR10 Dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,  # datasets를 통하여 CIFAR10데이터를 불러오고, 훈련 데이터 반환받음
                                             download=True, # 해당 경로에 CIFAR10데이터가 없다면 다운로드 받음
                                             transform=transforms.ToTensor()) # 현재 데이터를 파이토치 텐서로 변환

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,  # datasets를 통하여 CIFAR10데이터를 불러오고, 테스트 데이터 반환받음
                                            transform=transforms.ToTensor()) # 현재 데이터를 파이토치 텐서로 변환

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, # dataloader를 사용하여 train_dataset를 불러오고, 64의 배치사이즈 지정 및 셔플 활성화
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = data.DataLoader(dataset=test_dataset, # dataloader를 사용하여 test_dataset를 불러오고, 64의 배치사이즈 지정 및 셔플 비활성화
                              batch_size=batch_size,
                              shuffle=False)


class Net(nn.Module): # nn.Module의 상속을 받는 신경망 클래스 작성

    def __init__(self): # 3*32*32를 입력, 10을 출력으로 만들어주는 6개의 hidden layer를 만들어줌.
        super(Net, self).__init__()
        self.l1 = nn.Linear(3072, 2100) # input layer
        self.l2 = nn.Linear(2100, 1400) # hidden layer1
        self.l3 = nn.Linear(1400, 800) # hidden layer2
        self.l4 = nn.Linear(800, 400) # hidden layer3
        self.l5 = nn.Linear(400, 180) # hidden layer4
        self.l6 = nn.Linear(180, 10) # output layer

    def forward(self, x):
        x = x.view(-1, 3072)  # Flatten the data (n, 1, 28, 28)-> (n, 784), 데이터를 일렬로 펴 준다.
        x = F.relu(self.l1(x))  # 평탄화된 x를 선형함수 l1에 넣고, 활성화 relu함수에 또 넣은 뒤 결과를 x에 덮어씌움
        x = F.relu(self.l2(x))  # 평탄화된 x를 선형함수 l2에 넣고, 활성화 relu함수에 또 넣은 뒤 결과를 x에 덮어씌움
        x = F.relu(self.l3(x))  # 평탄화된 x를 선형함수 l3에 넣고, 활성화 relu함수에 또 넣은 뒤 결과를 x에 덮어씌움
        x = F.relu(self.l4(x))  # 평탄화된 x를 선형함수 l4에 넣고, 활성화 relu함수에 또 넣은 뒤 결과를 x에 덮어씌움
        x = F.relu(self.l5(x))  # 평탄화된 x를 선형함수 l5에 넣고, 활성화 relu함수에 또 넣은 뒤 결과를 x에 덮어씌움
        return self.l6(x) # l6를 출력



model = Net() # model에Net 클래스 적용
model.to(device)  # GPU가속 사용
criterion = nn.CrossEntropyLoss()   # criterion에 Cross EntropyLoss 사용
optimizer = optim.SGD(model.parameters(), lr=0.06, momentum=0.5)  # 최적화 함수로 경사하강법을 실행


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader): # train_loader를 이용하여 각각의 배치사이즈의 data, target를 불러옴
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)  # 예측값 output과 target의 손실 계산
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print('Train Epoch: {} | Batch Status: {}/{} ({:.0f}%) | Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(): # 학습이 완료된 모델에 test데이터셋을 적용하여 제대로 학습이 되었나 확인하는 과정
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device) # 연산에서 GPU가속 사용
        output = model(data)
        # sum up batch loss
        test_loss += criterion(output, target).item() # 테스트 손실 값을 계산
        # get the index of the max
        pred = output.data.max(1, keepdim=True)[1]  # output.data.max함수로 예측
        correct += pred.eq(target.data.view_as(pred)).cpu().sum() # 예측값과 타겟 데이터를 비교하여 얼마나 옳았는지 합을 계산(cpu로 연산)

    test_loss /= len(test_loader.dataset)
    print(f'===========================\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.0f}%)')  # 예측이 얼마나 맞았는지 정확도를 출력


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
Train Epoch: 1 | Batch Status: 0/50000 (0%) | Loss: 2.299517
Train Epoch: 1 | Batch Status: 640/50000 (1%) | Loss: 2.300468
Train Epoch: 1 | Batch Status: 1280/50000 (3%) | Loss: 2.300679
Train Epoch: 1 | Batch Status: 1920/50000 (4%) | Loss: 2.301934
Train Epoch: 1 | Batch Status: 2560/50000 (5%) | Loss: 2.305239
Train Epoch: 1 | Batch Status: 3200/50000 (6%) | Loss: 2.292032
Train Epoch: 1 | Batch Status: 3840/50000 (8%) | Loss: 2.305763
Train Epoch: 1 | Batch Status: 4480/50000 (9%) | Loss: 2.311584
Train Epoch: 1 | Batch Status: 5120/50000 (10%) | Loss: 2.300260
Train Epoch: 1 | Batch Status: 5760/50000 (12%) | Loss: 2.293859
Train Epoch: 1 | Batch Status: 6400/50000 (13%) | Loss: 2.298558
Train Epoch: 1 | Batch Status: 7040/50000 (14%) | Loss: 2.295980
Train Epoch: 1 | Batch Status: 7680/50000 (15%) | Loss: 2.288280
Train Epoch: 1 | Batch Status: 8320/50000 (17%) | Loss: 2.291692
Train Epoch: 1 | Batch Status: 8960/50000 (18%) | Loss: 2.294561
Train Epoch: 1 | Batch Status: 9600/50000 (19%) | Loss: 2.272400
Train Epoch: 1 | Batch Status: 10240/50000 (20%) | Loss: 2.240767
Train Epoch: 1 | Batch Status: 10880/50000 (22%) | Loss: 2.173904
Train Epoch: 1 | Batch Status: 11520/50000 (23%) | Loss: 2.185046
Train Epoch: 1 | Batch Status: 12160/50000 (24%) | Loss: 2.219388
Train Epoch: 1 | Batch Status: 12800/50000 (26%) | Loss: 2.069895
Train Epoch: 1 | Batch Status: 13440/50000 (27%) | Loss: 2.203301
Train Epoch: 1 | Batch Status: 14080/50000 (28%) | Loss: 2.108566
Train Epoch: 1 | Batch Status: 14720/50000 (29%) | Loss: 2.100180
Train Epoch: 1 | Batch Status: 15360/50000 (31%) | Loss: 2.072784
Train Epoch: 1 | Batch Status: 16000/50000 (32%) | Loss: 2.054780
Train Epoch: 1 | Batch Status: 16640/50000 (33%) | Loss: 2.071218
Train Epoch: 1 | Batch Status: 17280/50000 (35%) | Loss: 2.161995
Train Epoch: 1 | Batch Status: 17920/50000 (36%) | Loss: 2.232601
Train Epoch: 1 | Batch Status: 18560/50000 (37%) | Loss: 2.066640
Train Epoch: 1 | Batch Status: 19200/50000 (38%) | Loss: 2.076709
Train Epoch: 1 | Batch Status: 19840/50000 (40%) | Loss: 2.208882
Train Epoch: 1 | Batch Status: 20480/50000 (41%) | Loss: 2.065205
Train Epoch: 1 | Batch Status: 21120/50000 (42%) | Loss: 2.114381
Train Epoch: 1 | Batch Status: 21760/50000 (43%) | Loss: 2.077827
Train Epoch: 1 | Batch Status: 22400/50000 (45%) | Loss: 1.990838
Train Epoch: 1 | Batch Status: 23040/50000 (46%) | Loss: 2.066561
Train Epoch: 1 | Batch Status: 23680/50000 (47%) | Loss: 2.064729
Train Epoch: 1 | Batch Status: 24320/50000 (49%) | Loss: 2.085266
Train Epoch: 1 | Batch Status: 24960/50000 (50%) | Loss: 1.978016
Train Epoch: 1 | Batch Status: 25600/50000 (51%) | Loss: 1.927763
Train Epoch: 1 | Batch Status: 26240/50000 (52%) | Loss: 1.999255
Train Epoch: 1 | Batch Status: 26880/50000 (54%) | Loss: 2.025595
Train Epoch: 1 | Batch Status: 27520/50000 (55%) | Loss: 2.504839
Train Epoch: 1 | Batch Status: 28160/50000 (56%) | Loss: 1.893977
Train Epoch: 1 | Batch Status: 28800/50000 (58%) | Loss: 1.976406
Train Epoch: 1 | Batch Status: 29440/50000 (59%) | Loss: 1.884759
Train Epoch: 1 | Batch Status: 30080/50000 (60%) | Loss: 1.863095
Train Epoch: 1 | Batch Status: 30720/50000 (61%) | Loss: 2.153054
Train Epoch: 1 | Batch Status: 31360/50000 (63%) | Loss: 1.954919
Train Epoch: 1 | Batch Status: 32000/50000 (64%) | Loss: 1.882210
Train Epoch: 1 | Batch Status: 32640/50000 (65%) | Loss: 2.120567
Train Epoch: 1 | Batch Status: 33280/50000 (66%) | Loss: 1.950394
Train Epoch: 1 | Batch Status: 33920/50000 (68%) | Loss: 1.953321
Train Epoch: 1 | Batch Status: 34560/50000 (69%) | Loss: 1.944990
Train Epoch: 1 | Batch Status: 35200/50000 (70%) | Loss: 2.098306
Train Epoch: 1 | Batch Status: 35840/50000 (72%) | Loss: 2.041167
Train Epoch: 1 | Batch Status: 36480/50000 (73%) | Loss: 1.806518
Train Epoch: 1 | Batch Status: 37120/50000 (74%) | Loss: 1.878803
Train Epoch: 1 | Batch Status: 37760/50000 (75%) | Loss: 1.819255
Train Epoch: 1 | Batch Status: 38400/50000 (77%) | Loss: 2.038792
Train Epoch: 1 | Batch Status: 39040/50000 (78%) | Loss: 2.037681
Train Epoch: 1 | Batch Status: 39680/50000 (79%) | Loss: 1.960695
Train Epoch: 1 | Batch Status: 40320/50000 (81%) | Loss: 2.101614
Train Epoch: 1 | Batch Status: 40960/50000 (82%) | Loss: 1.845278
Train Epoch: 1 | Batch Status: 41600/50000 (83%) | Loss: 1.909683
Train Epoch: 1 | Batch Status: 42240/50000 (84%) | Loss: 1.938460
Train Epoch: 1 | Batch Status: 42880/50000 (86%) | Loss: 2.036551
Train Epoch: 1 | Batch Status: 43520/50000 (87%) | Loss: 1.952501
Train Epoch: 1 | Batch Status: 44160/50000 (88%) | Loss: 1.829201
Train Epoch: 1 | Batch Status: 44800/50000 (90%) | Loss: 1.876882
Train Epoch: 1 | Batch Status: 45440/50000 (91%) | Loss: 1.942494
Train Epoch: 1 | Batch Status: 46080/50000 (92%) | Loss: 1.943943
Train Epoch: 1 | Batch Status: 46720/50000 (93%) | Loss: 2.028497
Train Epoch: 1 | Batch Status: 47360/50000 (95%) | Loss: 1.906802
Train Epoch: 1 | Batch Status: 48000/50000 (96%) | Loss: 1.968020
Train Epoch: 1 | Batch Status: 48640/50000 (97%) | Loss: 1.885536
Train Epoch: 1 | Batch Status: 49280/50000 (98%) | Loss: 1.801048
Train Epoch: 1 | Batch Status: 49920/50000 (100%) | Loss: 1.916468
Training time: 0m 6s
===========================
Test set: Average loss: 0.0333, Accuracy: 2164/10000 (22%)
Testing time: 0m 7s
Train Epoch: 2 | Batch Status: 0/50000 (0%) | Loss: 2.096315
Train Epoch: 2 | Batch Status: 640/50000 (1%) | Loss: 1.679171
Train Epoch: 2 | Batch Status: 1280/50000 (3%) | Loss: 1.897215
Train Epoch: 2 | Batch Status: 1920/50000 (4%) | Loss: 1.849982
Train Epoch: 2 | Batch Status: 2560/50000 (5%) | Loss: 1.743164
Train Epoch: 2 | Batch Status: 3200/50000 (6%) | Loss: 1.914129
Train Epoch: 2 | Batch Status: 3840/50000 (8%) | Loss: 1.851234
Train Epoch: 2 | Batch Status: 4480/50000 (9%) | Loss: 1.770171
Train Epoch: 2 | Batch Status: 5120/50000 (10%) | Loss: 1.789148
Train Epoch: 2 | Batch Status: 5760/50000 (12%) | Loss: 1.966538
Train Epoch: 2 | Batch Status: 6400/50000 (13%) | Loss: 2.133669
Train Epoch: 2 | Batch Status: 7040/50000 (14%) | Loss: 1.719412
Train Epoch: 2 | Batch Status: 7680/50000 (15%) | Loss: 1.819665
Train Epoch: 2 | Batch Status: 8320/50000 (17%) | Loss: 1.859237
Train Epoch: 2 | Batch Status: 8960/50000 (18%) | Loss: 1.962066
Train Epoch: 2 | Batch Status: 9600/50000 (19%) | Loss: 1.655575
Train Epoch: 2 | Batch Status: 10240/50000 (20%) | Loss: 1.713628
Train Epoch: 2 | Batch Status: 10880/50000 (22%) | Loss: 1.775507
Train Epoch: 2 | Batch Status: 11520/50000 (23%) | Loss: 1.891528
Train Epoch: 2 | Batch Status: 12160/50000 (24%) | Loss: 1.792261
Train Epoch: 2 | Batch Status: 12800/50000 (26%) | Loss: 1.751327
Train Epoch: 2 | Batch Status: 13440/50000 (27%) | Loss: 1.720889
Train Epoch: 2 | Batch Status: 14080/50000 (28%) | Loss: 1.612605
Train Epoch: 2 | Batch Status: 14720/50000 (29%) | Loss: 1.769311
Train Epoch: 2 | Batch Status: 15360/50000 (31%) | Loss: 1.681345
Train Epoch: 2 | Batch Status: 16000/50000 (32%) | Loss: 1.905158
Train Epoch: 2 | Batch Status: 16640/50000 (33%) | Loss: 1.845866
Train Epoch: 2 | Batch Status: 17280/50000 (35%) | Loss: 1.833100
Train Epoch: 2 | Batch Status: 17920/50000 (36%) | Loss: 1.783708
Train Epoch: 2 | Batch Status: 18560/50000 (37%) | Loss: 1.847896
Train Epoch: 2 | Batch Status: 19200/50000 (38%) | Loss: 1.824806
Train Epoch: 2 | Batch Status: 19840/50000 (40%) | Loss: 1.852955
Train Epoch: 2 | Batch Status: 20480/50000 (41%) | Loss: 1.772444
Train Epoch: 2 | Batch Status: 21120/50000 (42%) | Loss: 1.851349
Train Epoch: 2 | Batch Status: 21760/50000 (43%) | Loss: 1.686065
Train Epoch: 2 | Batch Status: 22400/50000 (45%) | Loss: 1.839010
Train Epoch: 2 | Batch Status: 23040/50000 (46%) | Loss: 1.871813
Train Epoch: 2 | Batch Status: 23680/50000 (47%) | Loss: 1.583148
Train Epoch: 2 | Batch Status: 24320/50000 (49%) | Loss: 1.877020
Train Epoch: 2 | Batch Status: 24960/50000 (50%) | Loss: 1.939091
Train Epoch: 2 | Batch Status: 25600/50000 (51%) | Loss: 1.712692
Train Epoch: 2 | Batch Status: 26240/50000 (52%) | Loss: 1.611608
Train Epoch: 2 | Batch Status: 26880/50000 (54%) | Loss: 1.762259
Train Epoch: 2 | Batch Status: 27520/50000 (55%) | Loss: 1.802013
Train Epoch: 2 | Batch Status: 28160/50000 (56%) | Loss: 1.736924
Train Epoch: 2 | Batch Status: 28800/50000 (58%) | Loss: 1.823860
Train Epoch: 2 | Batch Status: 29440/50000 (59%) | Loss: 1.575830
Train Epoch: 2 | Batch Status: 30080/50000 (60%) | Loss: 1.666229
Train Epoch: 2 | Batch Status: 30720/50000 (61%) | Loss: 1.898795
Train Epoch: 2 | Batch Status: 31360/50000 (63%) | Loss: 1.860926
Train Epoch: 2 | Batch Status: 32000/50000 (64%) | Loss: 1.795831
Train Epoch: 2 | Batch Status: 32640/50000 (65%) | Loss: 1.694561
Train Epoch: 2 | Batch Status: 33280/50000 (66%) | Loss: 1.882877
Train Epoch: 2 | Batch Status: 33920/50000 (68%) | Loss: 1.705651
Train Epoch: 2 | Batch Status: 34560/50000 (69%) | Loss: 1.817083
Train Epoch: 2 | Batch Status: 35200/50000 (70%) | Loss: 1.605340
Train Epoch: 2 | Batch Status: 35840/50000 (72%) | Loss: 1.781129
Train Epoch: 2 | Batch Status: 36480/50000 (73%) | Loss: 1.700103
Train Epoch: 2 | Batch Status: 37120/50000 (74%) | Loss: 1.857195
Train Epoch: 2 | Batch Status: 37760/50000 (75%) | Loss: 1.529184
Train Epoch: 2 | Batch Status: 38400/50000 (77%) | Loss: 1.669655
Train Epoch: 2 | Batch Status: 39040/50000 (78%) | Loss: 1.922631
Train Epoch: 2 | Batch Status: 39680/50000 (79%) | Loss: 1.867904
Train Epoch: 2 | Batch Status: 40320/50000 (81%) | Loss: 1.798250
Train Epoch: 2 | Batch Status: 40960/50000 (82%) | Loss: 1.611841
Train Epoch: 2 | Batch Status: 41600/50000 (83%) | Loss: 1.616724
Train Epoch: 2 | Batch Status: 42240/50000 (84%) | Loss: 1.782451
Train Epoch: 2 | Batch Status: 42880/50000 (86%) | Loss: 1.770128
Train Epoch: 2 | Batch Status: 43520/50000 (87%) | Loss: 1.811634
Train Epoch: 2 | Batch Status: 44160/50000 (88%) | Loss: 1.741597
Train Epoch: 2 | Batch Status: 44800/50000 (90%) | Loss: 1.556338
Train Epoch: 2 | Batch Status: 45440/50000 (91%) | Loss: 1.796160
Train Epoch: 2 | Batch Status: 46080/50000 (92%) | Loss: 1.488561
Train Epoch: 2 | Batch Status: 46720/50000 (93%) | Loss: 1.734502
Train Epoch: 2 | Batch Status: 47360/50000 (95%) | Loss: 1.697291
Train Epoch: 2 | Batch Status: 48000/50000 (96%) | Loss: 1.707410
Train Epoch: 2 | Batch Status: 48640/50000 (97%) | Loss: 1.537662
Train Epoch: 2 | Batch Status: 49280/50000 (98%) | Loss: 1.711051
Train Epoch: 2 | Batch Status: 49920/50000 (100%) | Loss: 1.531702
Training time: 0m 6s
===========================
Test set: Average loss: 0.0299, Accuracy: 3091/10000 (31%)
Testing time: 0m 7s
Train Epoch: 3 | Batch Status: 0/50000 (0%) | Loss: 1.941687
Train Epoch: 3 | Batch Status: 640/50000 (1%) | Loss: 1.967440
Train Epoch: 3 | Batch Status: 1280/50000 (3%) | Loss: 1.669211
Train Epoch: 3 | Batch Status: 1920/50000 (4%) | Loss: 1.765765
Train Epoch: 3 | Batch Status: 2560/50000 (5%) | Loss: 1.841138
Train Epoch: 3 | Batch Status: 3200/50000 (6%) | Loss: 1.743640
Train Epoch: 3 | Batch Status: 3840/50000 (8%) | Loss: 1.749295
Train Epoch: 3 | Batch Status: 4480/50000 (9%) | Loss: 1.624228
Train Epoch: 3 | Batch Status: 5120/50000 (10%) | Loss: 1.740166
Train Epoch: 3 | Batch Status: 5760/50000 (12%) | Loss: 1.555166
Train Epoch: 3 | Batch Status: 6400/50000 (13%) | Loss: 1.694059
Train Epoch: 3 | Batch Status: 7040/50000 (14%) | Loss: 1.607652
Train Epoch: 3 | Batch Status: 7680/50000 (15%) | Loss: 1.780365
Train Epoch: 3 | Batch Status: 8320/50000 (17%) | Loss: 1.971246
Train Epoch: 3 | Batch Status: 8960/50000 (18%) | Loss: 1.614348
Train Epoch: 3 | Batch Status: 9600/50000 (19%) | Loss: 1.769212
Train Epoch: 3 | Batch Status: 10240/50000 (20%) | Loss: 1.768629
Train Epoch: 3 | Batch Status: 10880/50000 (22%) | Loss: 1.804370
Train Epoch: 3 | Batch Status: 11520/50000 (23%) | Loss: 1.604209
Train Epoch: 3 | Batch Status: 12160/50000 (24%) | Loss: 1.626698
Train Epoch: 3 | Batch Status: 12800/50000 (26%) | Loss: 1.549659
Train Epoch: 3 | Batch Status: 13440/50000 (27%) | Loss: 1.722854
Train Epoch: 3 | Batch Status: 14080/50000 (28%) | Loss: 1.710926
Train Epoch: 3 | Batch Status: 14720/50000 (29%) | Loss: 1.526875
Train Epoch: 3 | Batch Status: 15360/50000 (31%) | Loss: 1.624247
Train Epoch: 3 | Batch Status: 16000/50000 (32%) | Loss: 1.741360
Train Epoch: 3 | Batch Status: 16640/50000 (33%) | Loss: 1.599562
Train Epoch: 3 | Batch Status: 17280/50000 (35%) | Loss: 1.606123
Train Epoch: 3 | Batch Status: 17920/50000 (36%) | Loss: 1.666980
Train Epoch: 3 | Batch Status: 18560/50000 (37%) | Loss: 1.569715
Train Epoch: 3 | Batch Status: 19200/50000 (38%) | Loss: 1.680663
Train Epoch: 3 | Batch Status: 19840/50000 (40%) | Loss: 1.713187
Train Epoch: 3 | Batch Status: 20480/50000 (41%) | Loss: 1.843253
Train Epoch: 3 | Batch Status: 21120/50000 (42%) | Loss: 1.706529
Train Epoch: 3 | Batch Status: 21760/50000 (43%) | Loss: 1.500482
Train Epoch: 3 | Batch Status: 22400/50000 (45%) | Loss: 1.869967
Train Epoch: 3 | Batch Status: 23040/50000 (46%) | Loss: 1.514696
Train Epoch: 3 | Batch Status: 23680/50000 (47%) | Loss: 1.625786
Train Epoch: 3 | Batch Status: 24320/50000 (49%) | Loss: 1.631417
Train Epoch: 3 | Batch Status: 24960/50000 (50%) | Loss: 1.771277
Train Epoch: 3 | Batch Status: 25600/50000 (51%) | Loss: 1.863967
Train Epoch: 3 | Batch Status: 26240/50000 (52%) | Loss: 1.714336
Train Epoch: 3 | Batch Status: 26880/50000 (54%) | Loss: 1.940321
Train Epoch: 3 | Batch Status: 27520/50000 (55%) | Loss: 1.835164
Train Epoch: 3 | Batch Status: 28160/50000 (56%) | Loss: 1.809981
Train Epoch: 3 | Batch Status: 28800/50000 (58%) | Loss: 1.730466
Train Epoch: 3 | Batch Status: 29440/50000 (59%) | Loss: 1.575423
Train Epoch: 3 | Batch Status: 30080/50000 (60%) | Loss: 1.506882
Train Epoch: 3 | Batch Status: 30720/50000 (61%) | Loss: 1.892026
Train Epoch: 3 | Batch Status: 31360/50000 (63%) | Loss: 1.788136
Train Epoch: 3 | Batch Status: 32000/50000 (64%) | Loss: 1.519923
Train Epoch: 3 | Batch Status: 32640/50000 (65%) | Loss: 1.561067
Train Epoch: 3 | Batch Status: 33280/50000 (66%) | Loss: 1.756064
Train Epoch: 3 | Batch Status: 33920/50000 (68%) | Loss: 1.532147
Train Epoch: 3 | Batch Status: 34560/50000 (69%) | Loss: 1.593679
Train Epoch: 3 | Batch Status: 35200/50000 (70%) | Loss: 1.807589
Train Epoch: 3 | Batch Status: 35840/50000 (72%) | Loss: 1.800839
Train Epoch: 3 | Batch Status: 36480/50000 (73%) | Loss: 1.741605
Train Epoch: 3 | Batch Status: 37120/50000 (74%) | Loss: 1.426013
Train Epoch: 3 | Batch Status: 37760/50000 (75%) | Loss: 1.568876
Train Epoch: 3 | Batch Status: 38400/50000 (77%) | Loss: 1.776254
Train Epoch: 3 | Batch Status: 39040/50000 (78%) | Loss: 1.849732
Train Epoch: 3 | Batch Status: 39680/50000 (79%) | Loss: 1.730342
Train Epoch: 3 | Batch Status: 40320/50000 (81%) | Loss: 1.731348
Train Epoch: 3 | Batch Status: 40960/50000 (82%) | Loss: 1.774231
Train Epoch: 3 | Batch Status: 41600/50000 (83%) | Loss: 1.581743
Train Epoch: 3 | Batch Status: 42240/50000 (84%) | Loss: 1.587267
Train Epoch: 3 | Batch Status: 42880/50000 (86%) | Loss: 1.777035
Train Epoch: 3 | Batch Status: 43520/50000 (87%) | Loss: 1.876311
Train Epoch: 3 | Batch Status: 44160/50000 (88%) | Loss: 1.414631
Train Epoch: 3 | Batch Status: 44800/50000 (90%) | Loss: 1.557719
Train Epoch: 3 | Batch Status: 45440/50000 (91%) | Loss: 1.360352
Train Epoch: 3 | Batch Status: 46080/50000 (92%) | Loss: 1.692604
Train Epoch: 3 | Batch Status: 46720/50000 (93%) | Loss: 1.655812
Train Epoch: 3 | Batch Status: 47360/50000 (95%) | Loss: 1.683656
Train Epoch: 3 | Batch Status: 48000/50000 (96%) | Loss: 1.650380
Train Epoch: 3 | Batch Status: 48640/50000 (97%) | Loss: 1.485726
Train Epoch: 3 | Batch Status: 49280/50000 (98%) | Loss: 1.627633
Train Epoch: 3 | Batch Status: 49920/50000 (100%) | Loss: 1.588231
Training time: 0m 6s
===========================
Test set: Average loss: 0.0264, Accuracy: 3844/10000 (38%)
Testing time: 0m 7s
Train Epoch: 4 | Batch Status: 0/50000 (0%) | Loss: 1.691412
Train Epoch: 4 | Batch Status: 640/50000 (1%) | Loss: 1.582801
Train Epoch: 4 | Batch Status: 1280/50000 (3%) | Loss: 1.388239
Train Epoch: 4 | Batch Status: 1920/50000 (4%) | Loss: 1.696646
Train Epoch: 4 | Batch Status: 2560/50000 (5%) | Loss: 1.798628
Train Epoch: 4 | Batch Status: 3200/50000 (6%) | Loss: 1.425864
Train Epoch: 4 | Batch Status: 3840/50000 (8%) | Loss: 1.455094
Train Epoch: 4 | Batch Status: 4480/50000 (9%) | Loss: 1.594006
Train Epoch: 4 | Batch Status: 5120/50000 (10%) | Loss: 1.697764
Train Epoch: 4 | Batch Status: 5760/50000 (12%) | Loss: 1.548222
Train Epoch: 4 | Batch Status: 6400/50000 (13%) | Loss: 1.745641
Train Epoch: 4 | Batch Status: 7040/50000 (14%) | Loss: 1.614089
Train Epoch: 4 | Batch Status: 7680/50000 (15%) | Loss: 1.236207
Train Epoch: 4 | Batch Status: 8320/50000 (17%) | Loss: 1.489112
Train Epoch: 4 | Batch Status: 8960/50000 (18%) | Loss: 1.660745
Train Epoch: 4 | Batch Status: 9600/50000 (19%) | Loss: 1.834545
Train Epoch: 4 | Batch Status: 10240/50000 (20%) | Loss: 1.496146
Train Epoch: 4 | Batch Status: 10880/50000 (22%) | Loss: 1.814551
Train Epoch: 4 | Batch Status: 11520/50000 (23%) | Loss: 1.552045
Train Epoch: 4 | Batch Status: 12160/50000 (24%) | Loss: 1.693962
Train Epoch: 4 | Batch Status: 12800/50000 (26%) | Loss: 1.547879
Train Epoch: 4 | Batch Status: 13440/50000 (27%) | Loss: 1.666609
Train Epoch: 4 | Batch Status: 14080/50000 (28%) | Loss: 1.795274
Train Epoch: 4 | Batch Status: 14720/50000 (29%) | Loss: 1.442730
Train Epoch: 4 | Batch Status: 15360/50000 (31%) | Loss: 1.616932
Train Epoch: 4 | Batch Status: 16000/50000 (32%) | Loss: 1.702156
Train Epoch: 4 | Batch Status: 16640/50000 (33%) | Loss: 1.612272
Train Epoch: 4 | Batch Status: 17280/50000 (35%) | Loss: 1.763627
Train Epoch: 4 | Batch Status: 17920/50000 (36%) | Loss: 1.601490
Train Epoch: 4 | Batch Status: 18560/50000 (37%) | Loss: 1.721449
Train Epoch: 4 | Batch Status: 19200/50000 (38%) | Loss: 1.486655
Train Epoch: 4 | Batch Status: 19840/50000 (40%) | Loss: 1.787935
Train Epoch: 4 | Batch Status: 20480/50000 (41%) | Loss: 1.352563
Train Epoch: 4 | Batch Status: 21120/50000 (42%) | Loss: 1.537812
Train Epoch: 4 | Batch Status: 21760/50000 (43%) | Loss: 1.477536
Train Epoch: 4 | Batch Status: 22400/50000 (45%) | Loss: 1.760916
Train Epoch: 4 | Batch Status: 23040/50000 (46%) | Loss: 1.522571
Train Epoch: 4 | Batch Status: 23680/50000 (47%) | Loss: 1.590573
Train Epoch: 4 | Batch Status: 24320/50000 (49%) | Loss: 1.419204
Train Epoch: 4 | Batch Status: 24960/50000 (50%) | Loss: 1.648699
Train Epoch: 4 | Batch Status: 25600/50000 (51%) | Loss: 1.772016
Train Epoch: 4 | Batch Status: 26240/50000 (52%) | Loss: 1.864753
Train Epoch: 4 | Batch Status: 26880/50000 (54%) | Loss: 1.568771
Train Epoch: 4 | Batch Status: 27520/50000 (55%) | Loss: 1.490412
Train Epoch: 4 | Batch Status: 28160/50000 (56%) | Loss: 1.483050
Train Epoch: 4 | Batch Status: 28800/50000 (58%) | Loss: 1.634453
Train Epoch: 4 | Batch Status: 29440/50000 (59%) | Loss: 1.492311
Train Epoch: 4 | Batch Status: 30080/50000 (60%) | Loss: 1.633294
Train Epoch: 4 | Batch Status: 30720/50000 (61%) | Loss: 1.514984
Train Epoch: 4 | Batch Status: 31360/50000 (63%) | Loss: 1.478648
Train Epoch: 4 | Batch Status: 32000/50000 (64%) | Loss: 1.590866
Train Epoch: 4 | Batch Status: 32640/50000 (65%) | Loss: 1.643087
Train Epoch: 4 | Batch Status: 33280/50000 (66%) | Loss: 1.579193
Train Epoch: 4 | Batch Status: 33920/50000 (68%) | Loss: 1.525544
Train Epoch: 4 | Batch Status: 34560/50000 (69%) | Loss: 1.698010
Train Epoch: 4 | Batch Status: 35200/50000 (70%) | Loss: 1.463277
Train Epoch: 4 | Batch Status: 35840/50000 (72%) | Loss: 1.555004
Train Epoch: 4 | Batch Status: 36480/50000 (73%) | Loss: 1.703855
Train Epoch: 4 | Batch Status: 37120/50000 (74%) | Loss: 1.697889
Train Epoch: 4 | Batch Status: 37760/50000 (75%) | Loss: 1.576276
Train Epoch: 4 | Batch Status: 38400/50000 (77%) | Loss: 1.776843
Train Epoch: 4 | Batch Status: 39040/50000 (78%) | Loss: 1.605961
Train Epoch: 4 | Batch Status: 39680/50000 (79%) | Loss: 1.563850
Train Epoch: 4 | Batch Status: 40320/50000 (81%) | Loss: 1.684282
Train Epoch: 4 | Batch Status: 40960/50000 (82%) | Loss: 1.829818
Train Epoch: 4 | Batch Status: 41600/50000 (83%) | Loss: 1.694358
Train Epoch: 4 | Batch Status: 42240/50000 (84%) | Loss: 1.618863
Train Epoch: 4 | Batch Status: 42880/50000 (86%) | Loss: 1.712499
Train Epoch: 4 | Batch Status: 43520/50000 (87%) | Loss: 1.729941
Train Epoch: 4 | Batch Status: 44160/50000 (88%) | Loss: 1.643963
Train Epoch: 4 | Batch Status: 44800/50000 (90%) | Loss: 1.467618
Train Epoch: 4 | Batch Status: 45440/50000 (91%) | Loss: 1.444103
Train Epoch: 4 | Batch Status: 46080/50000 (92%) | Loss: 1.411100
Train Epoch: 4 | Batch Status: 46720/50000 (93%) | Loss: 1.526928
Train Epoch: 4 | Batch Status: 47360/50000 (95%) | Loss: 1.442900
Train Epoch: 4 | Batch Status: 48000/50000 (96%) | Loss: 1.494166
Train Epoch: 4 | Batch Status: 48640/50000 (97%) | Loss: 1.629100
Train Epoch: 4 | Batch Status: 49280/50000 (98%) | Loss: 1.415133
Train Epoch: 4 | Batch Status: 49920/50000 (100%) | Loss: 1.383252
Training time: 0m 6s
===========================
Test set: Average loss: 0.0269, Accuracy: 3999/10000 (40%)
Testing time: 0m 7s
Train Epoch: 5 | Batch Status: 0/50000 (0%) | Loss: 1.676877
Train Epoch: 5 | Batch Status: 640/50000 (1%) | Loss: 1.482231
Train Epoch: 5 | Batch Status: 1280/50000 (3%) | Loss: 1.440878
Train Epoch: 5 | Batch Status: 1920/50000 (4%) | Loss: 1.257478
Train Epoch: 5 | Batch Status: 2560/50000 (5%) | Loss: 1.333184
Train Epoch: 5 | Batch Status: 3200/50000 (6%) | Loss: 1.408106
Train Epoch: 5 | Batch Status: 3840/50000 (8%) | Loss: 1.551838
Train Epoch: 5 | Batch Status: 4480/50000 (9%) | Loss: 1.417377
Train Epoch: 5 | Batch Status: 5120/50000 (10%) | Loss: 1.490767
Train Epoch: 5 | Batch Status: 5760/50000 (12%) | Loss: 1.607751
Train Epoch: 5 | Batch Status: 6400/50000 (13%) | Loss: 1.568432
Train Epoch: 5 | Batch Status: 7040/50000 (14%) | Loss: 1.561242
Train Epoch: 5 | Batch Status: 7680/50000 (15%) | Loss: 1.454522
Train Epoch: 5 | Batch Status: 8320/50000 (17%) | Loss: 1.619561
Train Epoch: 5 | Batch Status: 8960/50000 (18%) | Loss: 1.731692
Train Epoch: 5 | Batch Status: 9600/50000 (19%) | Loss: 2.169490
Train Epoch: 5 | Batch Status: 10240/50000 (20%) | Loss: 1.372170
Train Epoch: 5 | Batch Status: 10880/50000 (22%) | Loss: 1.297798
Train Epoch: 5 | Batch Status: 11520/50000 (23%) | Loss: 1.643086
Train Epoch: 5 | Batch Status: 12160/50000 (24%) | Loss: 1.538720
Train Epoch: 5 | Batch Status: 12800/50000 (26%) | Loss: 1.566730
Train Epoch: 5 | Batch Status: 13440/50000 (27%) | Loss: 1.645712
Train Epoch: 5 | Batch Status: 14080/50000 (28%) | Loss: 1.684679
Train Epoch: 5 | Batch Status: 14720/50000 (29%) | Loss: 1.669632
Train Epoch: 5 | Batch Status: 15360/50000 (31%) | Loss: 1.658058
Train Epoch: 5 | Batch Status: 16000/50000 (32%) | Loss: 1.401034
Train Epoch: 5 | Batch Status: 16640/50000 (33%) | Loss: 1.555334
Train Epoch: 5 | Batch Status: 17280/50000 (35%) | Loss: 1.523523
Train Epoch: 5 | Batch Status: 17920/50000 (36%) | Loss: 1.543203
Train Epoch: 5 | Batch Status: 18560/50000 (37%) | Loss: 1.392461
Train Epoch: 5 | Batch Status: 19200/50000 (38%) | Loss: 1.619063
Train Epoch: 5 | Batch Status: 19840/50000 (40%) | Loss: 1.631482
Train Epoch: 5 | Batch Status: 20480/50000 (41%) | Loss: 1.632031
Train Epoch: 5 | Batch Status: 21120/50000 (42%) | Loss: 1.550088
Train Epoch: 5 | Batch Status: 21760/50000 (43%) | Loss: 1.654067
Train Epoch: 5 | Batch Status: 22400/50000 (45%) | Loss: 1.324377
Train Epoch: 5 | Batch Status: 23040/50000 (46%) | Loss: 1.577783
Train Epoch: 5 | Batch Status: 23680/50000 (47%) | Loss: 1.655534
Train Epoch: 5 | Batch Status: 24320/50000 (49%) | Loss: 1.542916
Train Epoch: 5 | Batch Status: 24960/50000 (50%) | Loss: 1.693799
Train Epoch: 5 | Batch Status: 25600/50000 (51%) | Loss: 1.771560
Train Epoch: 5 | Batch Status: 26240/50000 (52%) | Loss: 1.554001
Train Epoch: 5 | Batch Status: 26880/50000 (54%) | Loss: 1.581064
Train Epoch: 5 | Batch Status: 27520/50000 (55%) | Loss: 1.559958
Train Epoch: 5 | Batch Status: 28160/50000 (56%) | Loss: 1.599790
Train Epoch: 5 | Batch Status: 28800/50000 (58%) | Loss: 1.559062
Train Epoch: 5 | Batch Status: 29440/50000 (59%) | Loss: 1.506056
Train Epoch: 5 | Batch Status: 30080/50000 (60%) | Loss: 1.310394
Train Epoch: 5 | Batch Status: 30720/50000 (61%) | Loss: 1.593752
Train Epoch: 5 | Batch Status: 31360/50000 (63%) | Loss: 1.662733
Train Epoch: 5 | Batch Status: 32000/50000 (64%) | Loss: 1.483627
Train Epoch: 5 | Batch Status: 32640/50000 (65%) | Loss: 1.604412
Train Epoch: 5 | Batch Status: 33280/50000 (66%) | Loss: 1.388602
Train Epoch: 5 | Batch Status: 33920/50000 (68%) | Loss: 1.688298
Train Epoch: 5 | Batch Status: 34560/50000 (69%) | Loss: 1.585505
Train Epoch: 5 | Batch Status: 35200/50000 (70%) | Loss: 1.482901
Train Epoch: 5 | Batch Status: 35840/50000 (72%) | Loss: 1.232520
Train Epoch: 5 | Batch Status: 36480/50000 (73%) | Loss: 1.602426
Train Epoch: 5 | Batch Status: 37120/50000 (74%) | Loss: 1.693469
Train Epoch: 5 | Batch Status: 37760/50000 (75%) | Loss: 1.507045
Train Epoch: 5 | Batch Status: 38400/50000 (77%) | Loss: 1.409085
Train Epoch: 5 | Batch Status: 39040/50000 (78%) | Loss: 1.252456
Train Epoch: 5 | Batch Status: 39680/50000 (79%) | Loss: 1.420768
Train Epoch: 5 | Batch Status: 40320/50000 (81%) | Loss: 1.649795
Train Epoch: 5 | Batch Status: 40960/50000 (82%) | Loss: 1.359761
Train Epoch: 5 | Batch Status: 41600/50000 (83%) | Loss: 1.421044
Train Epoch: 5 | Batch Status: 42240/50000 (84%) | Loss: 1.250111
Train Epoch: 5 | Batch Status: 42880/50000 (86%) | Loss: 1.402361
Train Epoch: 5 | Batch Status: 43520/50000 (87%) | Loss: 1.544568
Train Epoch: 5 | Batch Status: 44160/50000 (88%) | Loss: 1.490769
Train Epoch: 5 | Batch Status: 44800/50000 (90%) | Loss: 1.551340
Train Epoch: 5 | Batch Status: 45440/50000 (91%) | Loss: 1.499369
Train Epoch: 5 | Batch Status: 46080/50000 (92%) | Loss: 1.508303
Train Epoch: 5 | Batch Status: 46720/50000 (93%) | Loss: 1.379766
Train Epoch: 5 | Batch Status: 47360/50000 (95%) | Loss: 1.732027
Train Epoch: 5 | Batch Status: 48000/50000 (96%) | Loss: 1.702514
Train Epoch: 5 | Batch Status: 48640/50000 (97%) | Loss: 1.555706
Train Epoch: 5 | Batch Status: 49280/50000 (98%) | Loss: 1.393483
Train Epoch: 5 | Batch Status: 49920/50000 (100%) | Loss: 1.470807
Training time: 0m 6s
===========================
Test set: Average loss: 0.0255, Accuracy: 4173/10000 (42%)
Testing time: 0m 7s
Train Epoch: 6 | Batch Status: 0/50000 (0%) | Loss: 1.556389
Train Epoch: 6 | Batch Status: 640/50000 (1%) | Loss: 1.356797
Train Epoch: 6 | Batch Status: 1280/50000 (3%) | Loss: 1.436408
Train Epoch: 6 | Batch Status: 1920/50000 (4%) | Loss: 1.497377
Train Epoch: 6 | Batch Status: 2560/50000 (5%) | Loss: 1.536885
Train Epoch: 6 | Batch Status: 3200/50000 (6%) | Loss: 1.737702
Train Epoch: 6 | Batch Status: 3840/50000 (8%) | Loss: 1.532842
Train Epoch: 6 | Batch Status: 4480/50000 (9%) | Loss: 1.558723
Train Epoch: 6 | Batch Status: 5120/50000 (10%) | Loss: 1.268031
Train Epoch: 6 | Batch Status: 5760/50000 (12%) | Loss: 1.686481
Train Epoch: 6 | Batch Status: 6400/50000 (13%) | Loss: 1.521176
Train Epoch: 6 | Batch Status: 7040/50000 (14%) | Loss: 1.575826
Train Epoch: 6 | Batch Status: 7680/50000 (15%) | Loss: 1.633774
Train Epoch: 6 | Batch Status: 8320/50000 (17%) | Loss: 1.392629
Train Epoch: 6 | Batch Status: 8960/50000 (18%) | Loss: 1.651455
Train Epoch: 6 | Batch Status: 9600/50000 (19%) | Loss: 1.557894
Train Epoch: 6 | Batch Status: 10240/50000 (20%) | Loss: 1.514265
Train Epoch: 6 | Batch Status: 10880/50000 (22%) | Loss: 1.483681
Train Epoch: 6 | Batch Status: 11520/50000 (23%) | Loss: 1.301394
Train Epoch: 6 | Batch Status: 12160/50000 (24%) | Loss: 1.284764
Train Epoch: 6 | Batch Status: 12800/50000 (26%) | Loss: 1.454719
Train Epoch: 6 | Batch Status: 13440/50000 (27%) | Loss: 1.291247
Train Epoch: 6 | Batch Status: 14080/50000 (28%) | Loss: 1.339309
Train Epoch: 6 | Batch Status: 14720/50000 (29%) | Loss: 1.603109
Train Epoch: 6 | Batch Status: 15360/50000 (31%) | Loss: 2.065801
Train Epoch: 6 | Batch Status: 16000/50000 (32%) | Loss: 1.545572
Train Epoch: 6 | Batch Status: 16640/50000 (33%) | Loss: 1.324990
Train Epoch: 6 | Batch Status: 17280/50000 (35%) | Loss: 1.464657
Train Epoch: 6 | Batch Status: 17920/50000 (36%) | Loss: 1.239005
Train Epoch: 6 | Batch Status: 18560/50000 (37%) | Loss: 1.429189
Train Epoch: 6 | Batch Status: 19200/50000 (38%) | Loss: 1.361007
Train Epoch: 6 | Batch Status: 19840/50000 (40%) | Loss: 1.520256
Train Epoch: 6 | Batch Status: 20480/50000 (41%) | Loss: 1.515309
Train Epoch: 6 | Batch Status: 21120/50000 (42%) | Loss: 1.499979
Train Epoch: 6 | Batch Status: 21760/50000 (43%) | Loss: 1.393446
Train Epoch: 6 | Batch Status: 22400/50000 (45%) | Loss: 1.529649
Train Epoch: 6 | Batch Status: 23040/50000 (46%) | Loss: 1.241107
Train Epoch: 6 | Batch Status: 23680/50000 (47%) | Loss: 1.296314
Train Epoch: 6 | Batch Status: 24320/50000 (49%) | Loss: 1.404848
Train Epoch: 6 | Batch Status: 24960/50000 (50%) | Loss: 1.396244
Train Epoch: 6 | Batch Status: 25600/50000 (51%) | Loss: 1.546030
Train Epoch: 6 | Batch Status: 26240/50000 (52%) | Loss: 1.470263
Train Epoch: 6 | Batch Status: 26880/50000 (54%) | Loss: 1.338559
Train Epoch: 6 | Batch Status: 27520/50000 (55%) | Loss: 1.299406
Train Epoch: 6 | Batch Status: 28160/50000 (56%) | Loss: 1.233299
Train Epoch: 6 | Batch Status: 28800/50000 (58%) | Loss: 1.455003
Train Epoch: 6 | Batch Status: 29440/50000 (59%) | Loss: 1.489842
Train Epoch: 6 | Batch Status: 30080/50000 (60%) | Loss: 1.429460
Train Epoch: 6 | Batch Status: 30720/50000 (61%) | Loss: 1.447391
Train Epoch: 6 | Batch Status: 31360/50000 (63%) | Loss: 1.449420
Train Epoch: 6 | Batch Status: 32000/50000 (64%) | Loss: 1.364775
Train Epoch: 6 | Batch Status: 32640/50000 (65%) | Loss: 1.552626
Train Epoch: 6 | Batch Status: 33280/50000 (66%) | Loss: 1.512000
Train Epoch: 6 | Batch Status: 33920/50000 (68%) | Loss: 1.692327
Train Epoch: 6 | Batch Status: 34560/50000 (69%) | Loss: 1.617597
Train Epoch: 6 | Batch Status: 35200/50000 (70%) | Loss: 1.556556
Train Epoch: 6 | Batch Status: 35840/50000 (72%) | Loss: 1.370552
Train Epoch: 6 | Batch Status: 36480/50000 (73%) | Loss: 1.310832
Train Epoch: 6 | Batch Status: 37120/50000 (74%) | Loss: 1.429541
Train Epoch: 6 | Batch Status: 37760/50000 (75%) | Loss: 1.607826
Train Epoch: 6 | Batch Status: 38400/50000 (77%) | Loss: 1.627092
Train Epoch: 6 | Batch Status: 39040/50000 (78%) | Loss: 1.247065
Train Epoch: 6 | Batch Status: 39680/50000 (79%) | Loss: 1.347268
Train Epoch: 6 | Batch Status: 40320/50000 (81%) | Loss: 1.469458
Train Epoch: 6 | Batch Status: 40960/50000 (82%) | Loss: 1.386815
Train Epoch: 6 | Batch Status: 41600/50000 (83%) | Loss: 1.663411
Train Epoch: 6 | Batch Status: 42240/50000 (84%) | Loss: 1.538257
Train Epoch: 6 | Batch Status: 42880/50000 (86%) | Loss: 1.427323
Train Epoch: 6 | Batch Status: 43520/50000 (87%) | Loss: 1.792851
Train Epoch: 6 | Batch Status: 44160/50000 (88%) | Loss: 1.750760
Train Epoch: 6 | Batch Status: 44800/50000 (90%) | Loss: 1.299665
Train Epoch: 6 | Batch Status: 45440/50000 (91%) | Loss: 1.513726
Train Epoch: 6 | Batch Status: 46080/50000 (92%) | Loss: 1.376908
Train Epoch: 6 | Batch Status: 46720/50000 (93%) | Loss: 1.382972
Train Epoch: 6 | Batch Status: 47360/50000 (95%) | Loss: 1.505129
Train Epoch: 6 | Batch Status: 48000/50000 (96%) | Loss: 1.422671
Train Epoch: 6 | Batch Status: 48640/50000 (97%) | Loss: 1.376371
Train Epoch: 6 | Batch Status: 49280/50000 (98%) | Loss: 1.597586
Train Epoch: 6 | Batch Status: 49920/50000 (100%) | Loss: 1.430652
Training time: 0m 6s
===========================
Test set: Average loss: 0.0377, Accuracy: 2790/10000 (28%)
Testing time: 0m 7s
Train Epoch: 7 | Batch Status: 0/50000 (0%) | Loss: 2.235112
Train Epoch: 7 | Batch Status: 640/50000 (1%) | Loss: 1.404958
Train Epoch: 7 | Batch Status: 1280/50000 (3%) | Loss: 1.524353
Train Epoch: 7 | Batch Status: 1920/50000 (4%) | Loss: 1.686804
Train Epoch: 7 | Batch Status: 2560/50000 (5%) | Loss: 1.551350
Train Epoch: 7 | Batch Status: 3200/50000 (6%) | Loss: 1.697255
Train Epoch: 7 | Batch Status: 3840/50000 (8%) | Loss: 1.510697
Train Epoch: 7 | Batch Status: 4480/50000 (9%) | Loss: 1.136926
Train Epoch: 7 | Batch Status: 5120/50000 (10%) | Loss: 1.529537
Train Epoch: 7 | Batch Status: 5760/50000 (12%) | Loss: 1.456782
Train Epoch: 7 | Batch Status: 6400/50000 (13%) | Loss: 1.466362
Train Epoch: 7 | Batch Status: 7040/50000 (14%) | Loss: 1.712739
Train Epoch: 7 | Batch Status: 7680/50000 (15%) | Loss: 1.469139
Train Epoch: 7 | Batch Status: 8320/50000 (17%) | Loss: 1.330179
Train Epoch: 7 | Batch Status: 8960/50000 (18%) | Loss: 1.355339
Train Epoch: 7 | Batch Status: 9600/50000 (19%) | Loss: 1.423444
Train Epoch: 7 | Batch Status: 10240/50000 (20%) | Loss: 1.328169
Train Epoch: 7 | Batch Status: 10880/50000 (22%) | Loss: 1.500465
Train Epoch: 7 | Batch Status: 11520/50000 (23%) | Loss: 1.437125
Train Epoch: 7 | Batch Status: 12160/50000 (24%) | Loss: 1.516627
Train Epoch: 7 | Batch Status: 12800/50000 (26%) | Loss: 1.402749
Train Epoch: 7 | Batch Status: 13440/50000 (27%) | Loss: 1.273979
Train Epoch: 7 | Batch Status: 14080/50000 (28%) | Loss: 1.461857
Train Epoch: 7 | Batch Status: 14720/50000 (29%) | Loss: 1.245481
Train Epoch: 7 | Batch Status: 15360/50000 (31%) | Loss: 1.573091
Train Epoch: 7 | Batch Status: 16000/50000 (32%) | Loss: 1.529924
Train Epoch: 7 | Batch Status: 16640/50000 (33%) | Loss: 1.419555
Train Epoch: 7 | Batch Status: 17280/50000 (35%) | Loss: 1.497225
Train Epoch: 7 | Batch Status: 17920/50000 (36%) | Loss: 1.426716
Train Epoch: 7 | Batch Status: 18560/50000 (37%) | Loss: 1.488684
Train Epoch: 7 | Batch Status: 19200/50000 (38%) | Loss: 1.461010
Train Epoch: 7 | Batch Status: 19840/50000 (40%) | Loss: 1.648192
Train Epoch: 7 | Batch Status: 20480/50000 (41%) | Loss: 1.733015
Train Epoch: 7 | Batch Status: 21120/50000 (42%) | Loss: 1.471855
Train Epoch: 7 | Batch Status: 21760/50000 (43%) | Loss: 1.436431
Train Epoch: 7 | Batch Status: 22400/50000 (45%) | Loss: 1.433311
Train Epoch: 7 | Batch Status: 23040/50000 (46%) | Loss: 1.337859
Train Epoch: 7 | Batch Status: 23680/50000 (47%) | Loss: 1.618099
Train Epoch: 7 | Batch Status: 24320/50000 (49%) | Loss: 1.511072
Train Epoch: 7 | Batch Status: 24960/50000 (50%) | Loss: 1.437682
Train Epoch: 7 | Batch Status: 25600/50000 (51%) | Loss: 1.390694
Train Epoch: 7 | Batch Status: 26240/50000 (52%) | Loss: 1.450957
Train Epoch: 7 | Batch Status: 26880/50000 (54%) | Loss: 1.509770
Train Epoch: 7 | Batch Status: 27520/50000 (55%) | Loss: 1.264803
Train Epoch: 7 | Batch Status: 28160/50000 (56%) | Loss: 1.294587
Train Epoch: 7 | Batch Status: 28800/50000 (58%) | Loss: 1.498724
Train Epoch: 7 | Batch Status: 29440/50000 (59%) | Loss: 1.632977
Train Epoch: 7 | Batch Status: 30080/50000 (60%) | Loss: 1.304818
Train Epoch: 7 | Batch Status: 30720/50000 (61%) | Loss: 1.453613
Train Epoch: 7 | Batch Status: 31360/50000 (63%) | Loss: 1.190889
Train Epoch: 7 | Batch Status: 32000/50000 (64%) | Loss: 1.415071
Train Epoch: 7 | Batch Status: 32640/50000 (65%) | Loss: 1.459224
Train Epoch: 7 | Batch Status: 33280/50000 (66%) | Loss: 1.300312
Train Epoch: 7 | Batch Status: 33920/50000 (68%) | Loss: 1.505705
Train Epoch: 7 | Batch Status: 34560/50000 (69%) | Loss: 1.435845
Train Epoch: 7 | Batch Status: 35200/50000 (70%) | Loss: 1.463218
Train Epoch: 7 | Batch Status: 35840/50000 (72%) | Loss: 1.342223
Train Epoch: 7 | Batch Status: 36480/50000 (73%) | Loss: 1.395258
Train Epoch: 7 | Batch Status: 37120/50000 (74%) | Loss: 1.514978
Train Epoch: 7 | Batch Status: 37760/50000 (75%) | Loss: 1.379577
Train Epoch: 7 | Batch Status: 38400/50000 (77%) | Loss: 1.309794
Train Epoch: 7 | Batch Status: 39040/50000 (78%) | Loss: 1.485358
Train Epoch: 7 | Batch Status: 39680/50000 (79%) | Loss: 1.312678
Train Epoch: 7 | Batch Status: 40320/50000 (81%) | Loss: 1.376140
Train Epoch: 7 | Batch Status: 40960/50000 (82%) | Loss: 1.532869
Train Epoch: 7 | Batch Status: 41600/50000 (83%) | Loss: 1.520629
Train Epoch: 7 | Batch Status: 42240/50000 (84%) | Loss: 1.252394
Train Epoch: 7 | Batch Status: 42880/50000 (86%) | Loss: 1.221799
Train Epoch: 7 | Batch Status: 43520/50000 (87%) | Loss: 1.395988
Train Epoch: 7 | Batch Status: 44160/50000 (88%) | Loss: 1.360057
Train Epoch: 7 | Batch Status: 44800/50000 (90%) | Loss: 1.466884
Train Epoch: 7 | Batch Status: 45440/50000 (91%) | Loss: 1.304819
Train Epoch: 7 | Batch Status: 46080/50000 (92%) | Loss: 1.268253
Train Epoch: 7 | Batch Status: 46720/50000 (93%) | Loss: 1.487207
Train Epoch: 7 | Batch Status: 47360/50000 (95%) | Loss: 1.604849
Train Epoch: 7 | Batch Status: 48000/50000 (96%) | Loss: 1.108755
Train Epoch: 7 | Batch Status: 48640/50000 (97%) | Loss: 1.394484
Train Epoch: 7 | Batch Status: 49280/50000 (98%) | Loss: 1.272394
Train Epoch: 7 | Batch Status: 49920/50000 (100%) | Loss: 1.494294
Training time: 0m 6s
===========================
Test set: Average loss: 0.0290, Accuracy: 3599/10000 (36%)
Testing time: 0m 7s
Train Epoch: 8 | Batch Status: 0/50000 (0%) | Loss: 2.131125
Train Epoch: 8 | Batch Status: 640/50000 (1%) | Loss: 1.328093
Train Epoch: 8 | Batch Status: 1280/50000 (3%) | Loss: 1.346120
Train Epoch: 8 | Batch Status: 1920/50000 (4%) | Loss: 1.318035
Train Epoch: 8 | Batch Status: 2560/50000 (5%) | Loss: 1.305111
Train Epoch: 8 | Batch Status: 3200/50000 (6%) | Loss: 1.278596
Train Epoch: 8 | Batch Status: 3840/50000 (8%) | Loss: 1.409067
Train Epoch: 8 | Batch Status: 4480/50000 (9%) | Loss: 1.366321
Train Epoch: 8 | Batch Status: 5120/50000 (10%) | Loss: 1.349123
Train Epoch: 8 | Batch Status: 5760/50000 (12%) | Loss: 1.260695
Train Epoch: 8 | Batch Status: 6400/50000 (13%) | Loss: 1.296293
Train Epoch: 8 | Batch Status: 7040/50000 (14%) | Loss: 1.431956
Train Epoch: 8 | Batch Status: 7680/50000 (15%) | Loss: 1.326072
Train Epoch: 8 | Batch Status: 8320/50000 (17%) | Loss: 1.366902
Train Epoch: 8 | Batch Status: 8960/50000 (18%) | Loss: 1.414825
Train Epoch: 8 | Batch Status: 9600/50000 (19%) | Loss: 1.423526
Train Epoch: 8 | Batch Status: 10240/50000 (20%) | Loss: 1.211761
Train Epoch: 8 | Batch Status: 10880/50000 (22%) | Loss: 1.222034
Train Epoch: 8 | Batch Status: 11520/50000 (23%) | Loss: 1.277112
Train Epoch: 8 | Batch Status: 12160/50000 (24%) | Loss: 1.174596
Train Epoch: 8 | Batch Status: 12800/50000 (26%) | Loss: 1.786816
Train Epoch: 8 | Batch Status: 13440/50000 (27%) | Loss: 1.207899
Train Epoch: 8 | Batch Status: 14080/50000 (28%) | Loss: 1.582820
Train Epoch: 8 | Batch Status: 14720/50000 (29%) | Loss: 1.691213
Train Epoch: 8 | Batch Status: 15360/50000 (31%) | Loss: 1.601482
Train Epoch: 8 | Batch Status: 16000/50000 (32%) | Loss: 1.209390
Train Epoch: 8 | Batch Status: 16640/50000 (33%) | Loss: 1.523407
Train Epoch: 8 | Batch Status: 17280/50000 (35%) | Loss: 1.339522
Train Epoch: 8 | Batch Status: 17920/50000 (36%) | Loss: 1.300441
Train Epoch: 8 | Batch Status: 18560/50000 (37%) | Loss: 1.272905
Train Epoch: 8 | Batch Status: 19200/50000 (38%) | Loss: 1.543629
Train Epoch: 8 | Batch Status: 19840/50000 (40%) | Loss: 1.394110
Train Epoch: 8 | Batch Status: 20480/50000 (41%) | Loss: 1.421983
Train Epoch: 8 | Batch Status: 21120/50000 (42%) | Loss: 1.266836
Train Epoch: 8 | Batch Status: 21760/50000 (43%) | Loss: 1.288144
Train Epoch: 8 | Batch Status: 22400/50000 (45%) | Loss: 1.425381
Train Epoch: 8 | Batch Status: 23040/50000 (46%) | Loss: 1.194624
Train Epoch: 8 | Batch Status: 23680/50000 (47%) | Loss: 1.419489
Train Epoch: 8 | Batch Status: 24320/50000 (49%) | Loss: 1.507066
Train Epoch: 8 | Batch Status: 24960/50000 (50%) | Loss: 1.515953
Train Epoch: 8 | Batch Status: 25600/50000 (51%) | Loss: 1.243771
Train Epoch: 8 | Batch Status: 26240/50000 (52%) | Loss: 1.376941
Train Epoch: 8 | Batch Status: 26880/50000 (54%) | Loss: 1.301289
Train Epoch: 8 | Batch Status: 27520/50000 (55%) | Loss: 1.441307
Train Epoch: 8 | Batch Status: 28160/50000 (56%) | Loss: 1.388929
Train Epoch: 8 | Batch Status: 28800/50000 (58%) | Loss: 1.546707
Train Epoch: 8 | Batch Status: 29440/50000 (59%) | Loss: 1.269847
Train Epoch: 8 | Batch Status: 30080/50000 (60%) | Loss: 1.424959
Train Epoch: 8 | Batch Status: 30720/50000 (61%) | Loss: 1.189798
Train Epoch: 8 | Batch Status: 31360/50000 (63%) | Loss: 1.144419
Train Epoch: 8 | Batch Status: 32000/50000 (64%) | Loss: 1.479954
Train Epoch: 8 | Batch Status: 32640/50000 (65%) | Loss: 1.395700
Train Epoch: 8 | Batch Status: 33280/50000 (66%) | Loss: 1.302770
Train Epoch: 8 | Batch Status: 33920/50000 (68%) | Loss: 1.415636
Train Epoch: 8 | Batch Status: 34560/50000 (69%) | Loss: 1.194924
Train Epoch: 8 | Batch Status: 35200/50000 (70%) | Loss: 1.496720
Train Epoch: 8 | Batch Status: 35840/50000 (72%) | Loss: 1.472807
Train Epoch: 8 | Batch Status: 36480/50000 (73%) | Loss: 1.408690
Train Epoch: 8 | Batch Status: 37120/50000 (74%) | Loss: 1.352785
Train Epoch: 8 | Batch Status: 37760/50000 (75%) | Loss: 1.485558
Train Epoch: 8 | Batch Status: 38400/50000 (77%) | Loss: 1.618062
Train Epoch: 8 | Batch Status: 39040/50000 (78%) | Loss: 1.374627
Train Epoch: 8 | Batch Status: 39680/50000 (79%) | Loss: 1.345251
Train Epoch: 8 | Batch Status: 40320/50000 (81%) | Loss: 1.369753
Train Epoch: 8 | Batch Status: 40960/50000 (82%) | Loss: 1.540117
Train Epoch: 8 | Batch Status: 41600/50000 (83%) | Loss: 1.268611
Train Epoch: 8 | Batch Status: 42240/50000 (84%) | Loss: 1.240932
Train Epoch: 8 | Batch Status: 42880/50000 (86%) | Loss: 1.527454
Train Epoch: 8 | Batch Status: 43520/50000 (87%) | Loss: 1.385543
Train Epoch: 8 | Batch Status: 44160/50000 (88%) | Loss: 1.365996
Train Epoch: 8 | Batch Status: 44800/50000 (90%) | Loss: 1.237201
Train Epoch: 8 | Batch Status: 45440/50000 (91%) | Loss: 1.863845
Train Epoch: 8 | Batch Status: 46080/50000 (92%) | Loss: 1.311201
Train Epoch: 8 | Batch Status: 46720/50000 (93%) | Loss: 1.348285
Train Epoch: 8 | Batch Status: 47360/50000 (95%) | Loss: 1.685446
Train Epoch: 8 | Batch Status: 48000/50000 (96%) | Loss: 1.320441
Train Epoch: 8 | Batch Status: 48640/50000 (97%) | Loss: 1.307990
Train Epoch: 8 | Batch Status: 49280/50000 (98%) | Loss: 1.553842
Train Epoch: 8 | Batch Status: 49920/50000 (100%) | Loss: 1.356524
Training time: 0m 6s
===========================
Test set: Average loss: 0.0231, Accuracy: 4829/10000 (48%)
Testing time: 0m 7s
Train Epoch: 9 | Batch Status: 0/50000 (0%) | Loss: 1.435900
Train Epoch: 9 | Batch Status: 640/50000 (1%) | Loss: 1.220934
Train Epoch: 9 | Batch Status: 1280/50000 (3%) | Loss: 1.233873
Train Epoch: 9 | Batch Status: 1920/50000 (4%) | Loss: 1.165329
Train Epoch: 9 | Batch Status: 2560/50000 (5%) | Loss: 1.331263
Train Epoch: 9 | Batch Status: 3200/50000 (6%) | Loss: 1.290336
Train Epoch: 9 | Batch Status: 3840/50000 (8%) | Loss: 1.341792
Train Epoch: 9 | Batch Status: 4480/50000 (9%) | Loss: 1.446344
Train Epoch: 9 | Batch Status: 5120/50000 (10%) | Loss: 1.524618
Train Epoch: 9 | Batch Status: 5760/50000 (12%) | Loss: 1.250522
Train Epoch: 9 | Batch Status: 6400/50000 (13%) | Loss: 1.449168
Train Epoch: 9 | Batch Status: 7040/50000 (14%) | Loss: 1.204307
Train Epoch: 9 | Batch Status: 7680/50000 (15%) | Loss: 1.448650
Train Epoch: 9 | Batch Status: 8320/50000 (17%) | Loss: 1.237021
Train Epoch: 9 | Batch Status: 8960/50000 (18%) | Loss: 1.326298
Train Epoch: 9 | Batch Status: 9600/50000 (19%) | Loss: 1.161639
Train Epoch: 9 | Batch Status: 10240/50000 (20%) | Loss: 1.507800
Train Epoch: 9 | Batch Status: 10880/50000 (22%) | Loss: 1.624806
Train Epoch: 9 | Batch Status: 11520/50000 (23%) | Loss: 1.326278
Train Epoch: 9 | Batch Status: 12160/50000 (24%) | Loss: 1.464116
Train Epoch: 9 | Batch Status: 12800/50000 (26%) | Loss: 1.190363
Train Epoch: 9 | Batch Status: 13440/50000 (27%) | Loss: 1.405617
Train Epoch: 9 | Batch Status: 14080/50000 (28%) | Loss: 1.367450
Train Epoch: 9 | Batch Status: 14720/50000 (29%) | Loss: 1.319947
Train Epoch: 9 | Batch Status: 15360/50000 (31%) | Loss: 1.526594
Train Epoch: 9 | Batch Status: 16000/50000 (32%) | Loss: 1.216047
Train Epoch: 9 | Batch Status: 16640/50000 (33%) | Loss: 1.429447
Train Epoch: 9 | Batch Status: 17280/50000 (35%) | Loss: 1.344801
Train Epoch: 9 | Batch Status: 17920/50000 (36%) | Loss: 1.352878
Train Epoch: 9 | Batch Status: 18560/50000 (37%) | Loss: 1.626915
Train Epoch: 9 | Batch Status: 19200/50000 (38%) | Loss: 1.501742
Train Epoch: 9 | Batch Status: 19840/50000 (40%) | Loss: 1.342941
Train Epoch: 9 | Batch Status: 20480/50000 (41%) | Loss: 1.371007
Train Epoch: 9 | Batch Status: 21120/50000 (42%) | Loss: 1.474267
Train Epoch: 9 | Batch Status: 21760/50000 (43%) | Loss: 1.276159
Train Epoch: 9 | Batch Status: 22400/50000 (45%) | Loss: 1.282512
Train Epoch: 9 | Batch Status: 23040/50000 (46%) | Loss: 1.367082
Train Epoch: 9 | Batch Status: 23680/50000 (47%) | Loss: 1.543043
Train Epoch: 9 | Batch Status: 24320/50000 (49%) | Loss: 1.132518
Train Epoch: 9 | Batch Status: 24960/50000 (50%) | Loss: 1.362114
Train Epoch: 9 | Batch Status: 25600/50000 (51%) | Loss: 1.268841
Train Epoch: 9 | Batch Status: 26240/50000 (52%) | Loss: 1.317015
Train Epoch: 9 | Batch Status: 26880/50000 (54%) | Loss: 1.311182
Train Epoch: 9 | Batch Status: 27520/50000 (55%) | Loss: 1.413452
Train Epoch: 9 | Batch Status: 28160/50000 (56%) | Loss: 1.618117
Train Epoch: 9 | Batch Status: 28800/50000 (58%) | Loss: 1.329808
Train Epoch: 9 | Batch Status: 29440/50000 (59%) | Loss: 1.309402
Train Epoch: 9 | Batch Status: 30080/50000 (60%) | Loss: 1.058638
Train Epoch: 9 | Batch Status: 30720/50000 (61%) | Loss: 1.410274
Train Epoch: 9 | Batch Status: 31360/50000 (63%) | Loss: 1.187111
Train Epoch: 9 | Batch Status: 32000/50000 (64%) | Loss: 1.245021
Train Epoch: 9 | Batch Status: 32640/50000 (65%) | Loss: 1.426194
Train Epoch: 9 | Batch Status: 33280/50000 (66%) | Loss: 1.517568
Train Epoch: 9 | Batch Status: 33920/50000 (68%) | Loss: 1.126224
Train Epoch: 9 | Batch Status: 34560/50000 (69%) | Loss: 1.524260
Train Epoch: 9 | Batch Status: 35200/50000 (70%) | Loss: 1.297792
Train Epoch: 9 | Batch Status: 35840/50000 (72%) | Loss: 1.231055
Train Epoch: 9 | Batch Status: 36480/50000 (73%) | Loss: 1.390111
Train Epoch: 9 | Batch Status: 37120/50000 (74%) | Loss: 1.361684
Train Epoch: 9 | Batch Status: 37760/50000 (75%) | Loss: 1.441717
Train Epoch: 9 | Batch Status: 38400/50000 (77%) | Loss: 1.543120
Train Epoch: 9 | Batch Status: 39040/50000 (78%) | Loss: 1.293520
Train Epoch: 9 | Batch Status: 39680/50000 (79%) | Loss: 1.185127
Train Epoch: 9 | Batch Status: 40320/50000 (81%) | Loss: 1.310617
Train Epoch: 9 | Batch Status: 40960/50000 (82%) | Loss: 1.286514
Train Epoch: 9 | Batch Status: 41600/50000 (83%) | Loss: 1.417680
Train Epoch: 9 | Batch Status: 42240/50000 (84%) | Loss: 1.491690
Train Epoch: 9 | Batch Status: 42880/50000 (86%) | Loss: 1.299429
Train Epoch: 9 | Batch Status: 43520/50000 (87%) | Loss: 1.277237
Train Epoch: 9 | Batch Status: 44160/50000 (88%) | Loss: 1.144357
Train Epoch: 9 | Batch Status: 44800/50000 (90%) | Loss: 1.343493
Train Epoch: 9 | Batch Status: 45440/50000 (91%) | Loss: 1.260510
Train Epoch: 9 | Batch Status: 46080/50000 (92%) | Loss: 1.407466
Train Epoch: 9 | Batch Status: 46720/50000 (93%) | Loss: 1.209958
Train Epoch: 9 | Batch Status: 47360/50000 (95%) | Loss: 1.159350
Train Epoch: 9 | Batch Status: 48000/50000 (96%) | Loss: 1.365543
Train Epoch: 9 | Batch Status: 48640/50000 (97%) | Loss: 1.265475
Train Epoch: 9 | Batch Status: 49280/50000 (98%) | Loss: 1.259151
Train Epoch: 9 | Batch Status: 49920/50000 (100%) | Loss: 1.426529
Training time: 0m 6s
===========================
Test set: Average loss: 0.0219, Accuracy: 5013/10000 (50%)
Testing time: 0m 7s
Total Time: 1m 2s
Model was trained on cuda!

Process finished with exit code 0

      
