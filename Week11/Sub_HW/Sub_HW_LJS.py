from __future__ import print_function
import argparse ## Script로 실행할 때 인자값에 따라 동작을 다르게 하기 위한 Module 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# Training settings
batch_size = 64

device = 'cuda' if torch.cuda.is_available() else 'cpu' # 사용가능하다면 연산에 GPU 사용 else CPU 사용 
print(f'Training MNIST Model on {device}\n{"=" * 44}')

# CIFAR10 Dataset -> Image size = 32 x 32 x 3 (RGB Channel)
train_dataset = datasets.CIFAR10(root='./data',train=True,transform=transforms.ToTensor(),download=True) 

test_dataset = datasets.CIFAR10(root='./data',train=False,transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_loader  = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # Conv2d(input channel, output channel, kernel size/filter size)
        # Convolutional Layer. In channel = 1, Out channel = 10, filter의 사이즈는 5x5
        # Default value => padding = 0 , stride = 1 
        
        # 입력 Channel = 3 
        # 출력 Channel = 20
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)   
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        
        self.mp = nn.MaxPool2d(2) 
        # Max Pooling kernel size = 2x2
        # Default value => padding = 0 , stride = 1 
        
        # 공간 데이터를 Flatten -> Dense Net 4 layers
        self.fc1 = nn.Linear(500, 320) 
        self.fc2 = nn.Linear(320, 160) 
        self.fc3 = nn.Linear(160, 80)
        self.fc4 = nn.Linear(80, 10)

    def forward(self, x):
      
        # torch.Size([64, 3, 32, 32]) -> Conv1
        # torch.Size([64, 10, 14, 14]) -> Conv2
        # torch.Size([64, 20, 5, 5]) MaxPool
        # torch.Size([64, 500]) -> Flaatten 

        # Convolution Net 
        in_size = x.size(0) # 들어온 data의 개수 -> data.shape = (n, 1, 28, 28) -> n = batch size 
        x = F.relu(self.mp(self.conv1(x))) # Convolution 수행 -> Maxpool -> relu activation function 1
        x = F.relu(self.mp(self.conv2(x))) # Convolution 수행 후 Maxpool -> relu activation function 2

        # Dense Net 
        x = x.view(in_size, -1)  # flatten the tensor
        x = self.fc1(x) # data를 flatten 한 후 Dense Net에 넣어줌 
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return F.log_softmax(x) # Calc Loss


model = Net()
model.to(device) # if GPU 사용한다면 GPU에 객체 올리기
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        data, target = data.to(device), target.to(device) # elements on device 
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
        data, target = data.to(device), target.to(device) # elements on device 
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
Training MNIST Model on cuda
============================================
Files already downloaded and verified
/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:66: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
Train Epoch: 1 [0/50000 (0%)]	Loss: 2.298575
Train Epoch: 1 [640/50000 (1%)]	Loss: 2.307077
Train Epoch: 1 [1280/50000 (3%)]	Loss: 2.289304
Train Epoch: 1 [1920/50000 (4%)]	Loss: 2.293433
Train Epoch: 1 [2560/50000 (5%)]	Loss: 2.300710
Train Epoch: 1 [3200/50000 (6%)]	Loss: 2.298124
Train Epoch: 1 [3840/50000 (8%)]	Loss: 2.302638
Train Epoch: 1 [4480/50000 (9%)]	Loss: 2.292657
Train Epoch: 1 [5120/50000 (10%)]	Loss: 2.287303
Train Epoch: 1 [5760/50000 (12%)]	Loss: 2.284164
Train Epoch: 1 [6400/50000 (13%)]	Loss: 2.279542
Train Epoch: 1 [7040/50000 (14%)]	Loss: 2.199678
Train Epoch: 1 [7680/50000 (15%)]	Loss: 2.266736
Train Epoch: 1 [8320/50000 (17%)]	Loss: 2.273515
Train Epoch: 1 [8960/50000 (18%)]	Loss: 2.230867
Train Epoch: 1 [9600/50000 (19%)]	Loss: 2.164472
Train Epoch: 1 [10240/50000 (20%)]	Loss: 2.118761
Train Epoch: 1 [10880/50000 (22%)]	Loss: 2.212328
Train Epoch: 1 [11520/50000 (23%)]	Loss: 2.150100
Train Epoch: 1 [12160/50000 (24%)]	Loss: 2.141227
Train Epoch: 1 [12800/50000 (26%)]	Loss: 2.029962
Train Epoch: 1 [13440/50000 (27%)]	Loss: 1.966288
Train Epoch: 1 [14080/50000 (28%)]	Loss: 2.066610
Train Epoch: 1 [14720/50000 (29%)]	Loss: 2.023261
Train Epoch: 1 [15360/50000 (31%)]	Loss: 2.130937
Train Epoch: 1 [16000/50000 (32%)]	Loss: 2.110325
Train Epoch: 1 [16640/50000 (33%)]	Loss: 1.931185
Train Epoch: 1 [17280/50000 (35%)]	Loss: 2.042412
Train Epoch: 1 [17920/50000 (36%)]	Loss: 1.920931
Train Epoch: 1 [18560/50000 (37%)]	Loss: 1.996001
Train Epoch: 1 [19200/50000 (38%)]	Loss: 1.985979
Train Epoch: 1 [19840/50000 (40%)]	Loss: 1.985362
Train Epoch: 1 [20480/50000 (41%)]	Loss: 1.914750
Train Epoch: 1 [21120/50000 (42%)]	Loss: 1.748849
Train Epoch: 1 [21760/50000 (43%)]	Loss: 2.005854
Train Epoch: 1 [22400/50000 (45%)]	Loss: 1.890685
Train Epoch: 1 [23040/50000 (46%)]	Loss: 1.737581
Train Epoch: 1 [23680/50000 (47%)]	Loss: 1.761891
Train Epoch: 1 [24320/50000 (49%)]	Loss: 1.820829
Train Epoch: 1 [24960/50000 (50%)]	Loss: 1.836653
Train Epoch: 1 [25600/50000 (51%)]	Loss: 1.651083
Train Epoch: 1 [26240/50000 (52%)]	Loss: 1.705983
Train Epoch: 1 [26880/50000 (54%)]	Loss: 1.883947
Train Epoch: 1 [27520/50000 (55%)]	Loss: 1.650115
Train Epoch: 1 [28160/50000 (56%)]	Loss: 1.902605
Train Epoch: 1 [28800/50000 (58%)]	Loss: 1.816437
Train Epoch: 1 [29440/50000 (59%)]	Loss: 1.752470
Train Epoch: 1 [30080/50000 (60%)]	Loss: 1.732029
Train Epoch: 1 [30720/50000 (61%)]	Loss: 1.810433
Train Epoch: 1 [31360/50000 (63%)]	Loss: 1.685409
Train Epoch: 1 [32000/50000 (64%)]	Loss: 1.748928
Train Epoch: 1 [32640/50000 (65%)]	Loss: 1.649912
Train Epoch: 1 [33280/50000 (66%)]	Loss: 1.658539
Train Epoch: 1 [33920/50000 (68%)]	Loss: 1.665292
Train Epoch: 1 [34560/50000 (69%)]	Loss: 1.740543
Train Epoch: 1 [35200/50000 (70%)]	Loss: 1.594183
Train Epoch: 1 [35840/50000 (72%)]	Loss: 1.733559
Train Epoch: 1 [36480/50000 (73%)]	Loss: 1.760588
Train Epoch: 1 [37120/50000 (74%)]	Loss: 1.569980
Train Epoch: 1 [37760/50000 (75%)]	Loss: 1.571229
Train Epoch: 1 [38400/50000 (77%)]	Loss: 1.512711
Train Epoch: 1 [39040/50000 (78%)]	Loss: 1.489175
Train Epoch: 1 [39680/50000 (79%)]	Loss: 1.592968
Train Epoch: 1 [40320/50000 (81%)]	Loss: 1.584887
Train Epoch: 1 [40960/50000 (82%)]	Loss: 1.565793
Train Epoch: 1 [41600/50000 (83%)]	Loss: 1.583306
Train Epoch: 1 [42240/50000 (84%)]	Loss: 1.646707
Train Epoch: 1 [42880/50000 (86%)]	Loss: 1.514448
Train Epoch: 1 [43520/50000 (87%)]	Loss: 1.645956
Train Epoch: 1 [44160/50000 (88%)]	Loss: 1.547440
Train Epoch: 1 [44800/50000 (90%)]	Loss: 1.501509
Train Epoch: 1 [45440/50000 (91%)]	Loss: 1.632078
Train Epoch: 1 [46080/50000 (92%)]	Loss: 1.693494
Train Epoch: 1 [46720/50000 (93%)]	Loss: 1.514431
Train Epoch: 1 [47360/50000 (95%)]	Loss: 1.657105
Train Epoch: 1 [48000/50000 (96%)]	Loss: 1.919564
Train Epoch: 1 [48640/50000 (97%)]	Loss: 1.634518
Train Epoch: 1 [49280/50000 (98%)]	Loss: 1.740317
Train Epoch: 1 [49920/50000 (100%)]	Loss: 1.634641
/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:96: UserWarning: volatile was removed and now has no effect. Use `with torch.no_grad():` instead.
/usr/local/lib/python3.7/dist-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='sum' instead.
  warnings.warn(warning.format(ret))

Test set: Average loss: 1.5063, Accuracy: 4512/10000 (45%)

Train Epoch: 2 [0/50000 (0%)]	Loss: 1.516784
Train Epoch: 2 [640/50000 (1%)]	Loss: 1.567662
Train Epoch: 2 [1280/50000 (3%)]	Loss: 1.540935
Train Epoch: 2 [1920/50000 (4%)]	Loss: 1.659188
Train Epoch: 2 [2560/50000 (5%)]	Loss: 1.694610
Train Epoch: 2 [3200/50000 (6%)]	Loss: 1.424187
Train Epoch: 2 [3840/50000 (8%)]	Loss: 1.352432
Train Epoch: 2 [4480/50000 (9%)]	Loss: 1.610740
Train Epoch: 2 [5120/50000 (10%)]	Loss: 1.495662
Train Epoch: 2 [5760/50000 (12%)]	Loss: 1.640773
Train Epoch: 2 [6400/50000 (13%)]	Loss: 1.452551
Train Epoch: 2 [7040/50000 (14%)]	Loss: 1.278109
Train Epoch: 2 [7680/50000 (15%)]	Loss: 1.570006
Train Epoch: 2 [8320/50000 (17%)]	Loss: 1.557753
Train Epoch: 2 [8960/50000 (18%)]	Loss: 1.551540
Train Epoch: 2 [9600/50000 (19%)]	Loss: 1.904353
Train Epoch: 2 [10240/50000 (20%)]	Loss: 1.658939
Train Epoch: 2 [10880/50000 (22%)]	Loss: 1.596680
Train Epoch: 2 [11520/50000 (23%)]	Loss: 1.573635
Train Epoch: 2 [12160/50000 (24%)]	Loss: 1.365132
Train Epoch: 2 [12800/50000 (26%)]	Loss: 1.345130
Train Epoch: 2 [13440/50000 (27%)]	Loss: 1.645405
Train Epoch: 2 [14080/50000 (28%)]	Loss: 1.404458
Train Epoch: 2 [14720/50000 (29%)]	Loss: 1.739829
Train Epoch: 2 [15360/50000 (31%)]	Loss: 1.466735
Train Epoch: 2 [16000/50000 (32%)]	Loss: 1.575112
Train Epoch: 2 [16640/50000 (33%)]	Loss: 1.386683
Train Epoch: 2 [17280/50000 (35%)]	Loss: 1.456824
Train Epoch: 2 [17920/50000 (36%)]	Loss: 1.457799
Train Epoch: 2 [18560/50000 (37%)]	Loss: 1.617460
Train Epoch: 2 [19200/50000 (38%)]	Loss: 1.642541
Train Epoch: 2 [19840/50000 (40%)]	Loss: 1.290255
Train Epoch: 2 [20480/50000 (41%)]	Loss: 1.636489
Train Epoch: 2 [21120/50000 (42%)]	Loss: 1.258403
Train Epoch: 2 [21760/50000 (43%)]	Loss: 1.472773
Train Epoch: 2 [22400/50000 (45%)]	Loss: 1.384991
Train Epoch: 2 [23040/50000 (46%)]	Loss: 1.418829
Train Epoch: 2 [23680/50000 (47%)]	Loss: 1.697123
Train Epoch: 2 [24320/50000 (49%)]	Loss: 1.311564
Train Epoch: 2 [24960/50000 (50%)]	Loss: 1.587999
Train Epoch: 2 [25600/50000 (51%)]	Loss: 1.450225
Train Epoch: 2 [26240/50000 (52%)]	Loss: 1.377094
Train Epoch: 2 [26880/50000 (54%)]	Loss: 1.627616
Train Epoch: 2 [27520/50000 (55%)]	Loss: 1.540498
Train Epoch: 2 [28160/50000 (56%)]	Loss: 1.513749
Train Epoch: 2 [28800/50000 (58%)]	Loss: 1.460629
Train Epoch: 2 [29440/50000 (59%)]	Loss: 1.362943
Train Epoch: 2 [30080/50000 (60%)]	Loss: 1.345564
Train Epoch: 2 [30720/50000 (61%)]	Loss: 1.416271
Train Epoch: 2 [31360/50000 (63%)]	Loss: 1.545520
Train Epoch: 2 [32000/50000 (64%)]	Loss: 1.508822
Train Epoch: 2 [32640/50000 (65%)]	Loss: 1.514776
Train Epoch: 2 [33280/50000 (66%)]	Loss: 1.383069
Train Epoch: 2 [33920/50000 (68%)]	Loss: 1.506771
Train Epoch: 2 [34560/50000 (69%)]	Loss: 1.448759
Train Epoch: 2 [35200/50000 (70%)]	Loss: 1.485468
Train Epoch: 2 [35840/50000 (72%)]	Loss: 1.484903
Train Epoch: 2 [36480/50000 (73%)]	Loss: 1.541163
Train Epoch: 2 [37120/50000 (74%)]	Loss: 1.346077
Train Epoch: 2 [37760/50000 (75%)]	Loss: 1.503474
Train Epoch: 2 [38400/50000 (77%)]	Loss: 1.303354
Train Epoch: 2 [39040/50000 (78%)]	Loss: 1.456706
Train Epoch: 2 [39680/50000 (79%)]	Loss: 1.539912
Train Epoch: 2 [40320/50000 (81%)]	Loss: 1.321987
Train Epoch: 2 [40960/50000 (82%)]	Loss: 1.366678
Train Epoch: 2 [41600/50000 (83%)]	Loss: 1.395025
Train Epoch: 2 [42240/50000 (84%)]	Loss: 1.356640
Train Epoch: 2 [42880/50000 (86%)]	Loss: 1.103971
Train Epoch: 2 [43520/50000 (87%)]	Loss: 1.390252
Train Epoch: 2 [44160/50000 (88%)]	Loss: 1.374245
Train Epoch: 2 [44800/50000 (90%)]	Loss: 1.417198
Train Epoch: 2 [45440/50000 (91%)]	Loss: 1.298706
Train Epoch: 2 [46080/50000 (92%)]	Loss: 1.506078
Train Epoch: 2 [46720/50000 (93%)]	Loss: 1.583010
Train Epoch: 2 [47360/50000 (95%)]	Loss: 1.299226
Train Epoch: 2 [48000/50000 (96%)]	Loss: 1.266473
Train Epoch: 2 [48640/50000 (97%)]	Loss: 1.200770
Train Epoch: 2 [49280/50000 (98%)]	Loss: 1.128456
Train Epoch: 2 [49920/50000 (100%)]	Loss: 1.194557

Test set: Average loss: 1.3639, Accuracy: 5121/10000 (51%)

Train Epoch: 3 [0/50000 (0%)]	Loss: 1.286137
Train Epoch: 3 [640/50000 (1%)]	Loss: 1.861170
Train Epoch: 3 [1280/50000 (3%)]	Loss: 1.427312
Train Epoch: 3 [1920/50000 (4%)]	Loss: 1.374628
Train Epoch: 3 [2560/50000 (5%)]	Loss: 1.347711
Train Epoch: 3 [3200/50000 (6%)]	Loss: 1.292999
Train Epoch: 3 [3840/50000 (8%)]	Loss: 1.233366
Train Epoch: 3 [4480/50000 (9%)]	Loss: 1.445532
Train Epoch: 3 [5120/50000 (10%)]	Loss: 1.454744
Train Epoch: 3 [5760/50000 (12%)]	Loss: 1.564921
Train Epoch: 3 [6400/50000 (13%)]	Loss: 1.266654
Train Epoch: 3 [7040/50000 (14%)]	Loss: 1.412335
Train Epoch: 3 [7680/50000 (15%)]	Loss: 1.364053
Train Epoch: 3 [8320/50000 (17%)]	Loss: 1.357884
Train Epoch: 3 [8960/50000 (18%)]	Loss: 1.499291
Train Epoch: 3 [9600/50000 (19%)]	Loss: 1.547387
Train Epoch: 3 [10240/50000 (20%)]	Loss: 1.436965
Train Epoch: 3 [10880/50000 (22%)]	Loss: 1.422848
Train Epoch: 3 [11520/50000 (23%)]	Loss: 1.178266
Train Epoch: 3 [12160/50000 (24%)]	Loss: 1.471513
Train Epoch: 3 [12800/50000 (26%)]	Loss: 1.658830
Train Epoch: 3 [13440/50000 (27%)]	Loss: 1.453737
Train Epoch: 3 [14080/50000 (28%)]	Loss: 1.249596
Train Epoch: 3 [14720/50000 (29%)]	Loss: 1.299292
Train Epoch: 3 [15360/50000 (31%)]	Loss: 1.479001
Train Epoch: 3 [16000/50000 (32%)]	Loss: 1.168809
Train Epoch: 3 [16640/50000 (33%)]	Loss: 1.228912
Train Epoch: 3 [17280/50000 (35%)]	Loss: 1.462106
Train Epoch: 3 [17920/50000 (36%)]	Loss: 1.186872
Train Epoch: 3 [18560/50000 (37%)]	Loss: 1.472797
Train Epoch: 3 [19200/50000 (38%)]	Loss: 1.579698
Train Epoch: 3 [19840/50000 (40%)]	Loss: 1.500318
Train Epoch: 3 [20480/50000 (41%)]	Loss: 1.452729
Train Epoch: 3 [21120/50000 (42%)]	Loss: 1.141037
Train Epoch: 3 [21760/50000 (43%)]	Loss: 1.367054
Train Epoch: 3 [22400/50000 (45%)]	Loss: 1.321825
Train Epoch: 3 [23040/50000 (46%)]	Loss: 1.339789
Train Epoch: 3 [23680/50000 (47%)]	Loss: 1.545305
Train Epoch: 3 [24320/50000 (49%)]	Loss: 1.333914
Train Epoch: 3 [24960/50000 (50%)]	Loss: 1.445718
Train Epoch: 3 [25600/50000 (51%)]	Loss: 1.074965
Train Epoch: 3 [26240/50000 (52%)]	Loss: 1.162862
Train Epoch: 3 [26880/50000 (54%)]	Loss: 1.226813
Train Epoch: 3 [27520/50000 (55%)]	Loss: 1.109256
Train Epoch: 3 [28160/50000 (56%)]	Loss: 1.271182
Train Epoch: 3 [28800/50000 (58%)]	Loss: 1.347321
Train Epoch: 3 [29440/50000 (59%)]	Loss: 1.370171
Train Epoch: 3 [30080/50000 (60%)]	Loss: 1.220575
Train Epoch: 3 [30720/50000 (61%)]	Loss: 1.293223
Train Epoch: 3 [31360/50000 (63%)]	Loss: 1.734954
Train Epoch: 3 [32000/50000 (64%)]	Loss: 1.558850
Train Epoch: 3 [32640/50000 (65%)]	Loss: 1.273089
Train Epoch: 3 [33280/50000 (66%)]	Loss: 1.472311
Train Epoch: 3 [33920/50000 (68%)]	Loss: 1.288296
Train Epoch: 3 [34560/50000 (69%)]	Loss: 1.349340
Train Epoch: 3 [35200/50000 (70%)]	Loss: 1.119847
Train Epoch: 3 [35840/50000 (72%)]	Loss: 1.492574
Train Epoch: 3 [36480/50000 (73%)]	Loss: 1.345969
Train Epoch: 3 [37120/50000 (74%)]	Loss: 1.171152
Train Epoch: 3 [37760/50000 (75%)]	Loss: 1.301359
Train Epoch: 3 [38400/50000 (77%)]	Loss: 1.508280
Train Epoch: 3 [39040/50000 (78%)]	Loss: 1.025432
Train Epoch: 3 [39680/50000 (79%)]	Loss: 1.430155
Train Epoch: 3 [40320/50000 (81%)]	Loss: 1.292401
Train Epoch: 3 [40960/50000 (82%)]	Loss: 1.488065
Train Epoch: 3 [41600/50000 (83%)]	Loss: 1.080935
Train Epoch: 3 [42240/50000 (84%)]	Loss: 1.119708
Train Epoch: 3 [42880/50000 (86%)]	Loss: 1.442471
Train Epoch: 3 [43520/50000 (87%)]	Loss: 1.368169
Train Epoch: 3 [44160/50000 (88%)]	Loss: 1.371059
Train Epoch: 3 [44800/50000 (90%)]	Loss: 1.274222
Train Epoch: 3 [45440/50000 (91%)]	Loss: 1.116640
Train Epoch: 3 [46080/50000 (92%)]	Loss: 1.196725
Train Epoch: 3 [46720/50000 (93%)]	Loss: 1.104247
Train Epoch: 3 [47360/50000 (95%)]	Loss: 1.403900
Train Epoch: 3 [48000/50000 (96%)]	Loss: 1.324739
Train Epoch: 3 [48640/50000 (97%)]	Loss: 1.218357
Train Epoch: 3 [49280/50000 (98%)]	Loss: 1.425888
Train Epoch: 3 [49920/50000 (100%)]	Loss: 1.280342

Test set: Average loss: 1.2667, Accuracy: 5470/10000 (55%)

Train Epoch: 4 [0/50000 (0%)]	Loss: 1.551596
Train Epoch: 4 [640/50000 (1%)]	Loss: 1.272130
Train Epoch: 4 [1280/50000 (3%)]	Loss: 1.287979
Train Epoch: 4 [1920/50000 (4%)]	Loss: 1.086050
Train Epoch: 4 [2560/50000 (5%)]	Loss: 1.220035
Train Epoch: 4 [3200/50000 (6%)]	Loss: 1.019182
Train Epoch: 4 [3840/50000 (8%)]	Loss: 1.160767
Train Epoch: 4 [4480/50000 (9%)]	Loss: 1.187638
Train Epoch: 4 [5120/50000 (10%)]	Loss: 1.511570
Train Epoch: 4 [5760/50000 (12%)]	Loss: 1.298180
Train Epoch: 4 [6400/50000 (13%)]	Loss: 1.140634
Train Epoch: 4 [7040/50000 (14%)]	Loss: 1.457679
Train Epoch: 4 [7680/50000 (15%)]	Loss: 1.189267
Train Epoch: 4 [8320/50000 (17%)]	Loss: 1.495416
Train Epoch: 4 [8960/50000 (18%)]	Loss: 1.104122
Train Epoch: 4 [9600/50000 (19%)]	Loss: 1.405559
Train Epoch: 4 [10240/50000 (20%)]	Loss: 1.329012
Train Epoch: 4 [10880/50000 (22%)]	Loss: 1.268432
Train Epoch: 4 [11520/50000 (23%)]	Loss: 1.430360
Train Epoch: 4 [12160/50000 (24%)]	Loss: 1.283281
Train Epoch: 4 [12800/50000 (26%)]	Loss: 1.311596
Train Epoch: 4 [13440/50000 (27%)]	Loss: 1.225226
Train Epoch: 4 [14080/50000 (28%)]	Loss: 1.311497
Train Epoch: 4 [14720/50000 (29%)]	Loss: 1.125928
Train Epoch: 4 [15360/50000 (31%)]	Loss: 1.654660
Train Epoch: 4 [16000/50000 (32%)]	Loss: 1.212145
Train Epoch: 4 [16640/50000 (33%)]	Loss: 1.105888
Train Epoch: 4 [17280/50000 (35%)]	Loss: 1.110732
Train Epoch: 4 [17920/50000 (36%)]	Loss: 1.139845
Train Epoch: 4 [18560/50000 (37%)]	Loss: 1.112032
Train Epoch: 4 [19200/50000 (38%)]	Loss: 1.274921
Train Epoch: 4 [19840/50000 (40%)]	Loss: 1.244441
Train Epoch: 4 [20480/50000 (41%)]	Loss: 1.270907
Train Epoch: 4 [21120/50000 (42%)]	Loss: 1.340999
Train Epoch: 4 [21760/50000 (43%)]	Loss: 1.582678
Train Epoch: 4 [22400/50000 (45%)]	Loss: 1.312641
Train Epoch: 4 [23040/50000 (46%)]	Loss: 1.142849
Train Epoch: 4 [23680/50000 (47%)]	Loss: 1.059750
Train Epoch: 4 [24320/50000 (49%)]	Loss: 1.056077
Train Epoch: 4 [24960/50000 (50%)]	Loss: 1.142960
Train Epoch: 4 [25600/50000 (51%)]	Loss: 1.312921
Train Epoch: 4 [26240/50000 (52%)]	Loss: 1.113376
Train Epoch: 4 [26880/50000 (54%)]	Loss: 1.382323
Train Epoch: 4 [27520/50000 (55%)]	Loss: 1.388389
Train Epoch: 4 [28160/50000 (56%)]	Loss: 1.396634
Train Epoch: 4 [28800/50000 (58%)]	Loss: 1.219810
Train Epoch: 4 [29440/50000 (59%)]	Loss: 1.237197
Train Epoch: 4 [30080/50000 (60%)]	Loss: 1.175979
Train Epoch: 4 [30720/50000 (61%)]	Loss: 0.876742
Train Epoch: 4 [31360/50000 (63%)]	Loss: 1.158206
Train Epoch: 4 [32000/50000 (64%)]	Loss: 0.875567
Train Epoch: 4 [32640/50000 (65%)]	Loss: 1.356879
Train Epoch: 4 [33280/50000 (66%)]	Loss: 1.130873
Train Epoch: 4 [33920/50000 (68%)]	Loss: 1.170399
Train Epoch: 4 [34560/50000 (69%)]	Loss: 1.152381
Train Epoch: 4 [35200/50000 (70%)]	Loss: 1.139454
Train Epoch: 4 [35840/50000 (72%)]	Loss: 1.170409
Train Epoch: 4 [36480/50000 (73%)]	Loss: 1.220039
Train Epoch: 4 [37120/50000 (74%)]	Loss: 1.342333
Train Epoch: 4 [37760/50000 (75%)]	Loss: 1.147738
Train Epoch: 4 [38400/50000 (77%)]	Loss: 1.301469
Train Epoch: 4 [39040/50000 (78%)]	Loss: 1.350271
Train Epoch: 4 [39680/50000 (79%)]	Loss: 1.107165
Train Epoch: 4 [40320/50000 (81%)]	Loss: 1.057466
Train Epoch: 4 [40960/50000 (82%)]	Loss: 1.063823
Train Epoch: 4 [41600/50000 (83%)]	Loss: 1.322542
Train Epoch: 4 [42240/50000 (84%)]	Loss: 1.278366
Train Epoch: 4 [42880/50000 (86%)]	Loss: 1.337676
Train Epoch: 4 [43520/50000 (87%)]	Loss: 1.409284
Train Epoch: 4 [44160/50000 (88%)]	Loss: 1.291396
Train Epoch: 4 [44800/50000 (90%)]	Loss: 1.244445
Train Epoch: 4 [45440/50000 (91%)]	Loss: 1.280126
Train Epoch: 4 [46080/50000 (92%)]	Loss: 1.254475
Train Epoch: 4 [46720/50000 (93%)]	Loss: 1.356391
Train Epoch: 4 [47360/50000 (95%)]	Loss: 1.220675
Train Epoch: 4 [48000/50000 (96%)]	Loss: 1.065977
Train Epoch: 4 [48640/50000 (97%)]	Loss: 1.294039
Train Epoch: 4 [49280/50000 (98%)]	Loss: 1.119278
Train Epoch: 4 [49920/50000 (100%)]	Loss: 1.104686

Test set: Average loss: 1.2314, Accuracy: 5583/10000 (56%)

Train Epoch: 5 [0/50000 (0%)]	Loss: 0.995772
Train Epoch: 5 [640/50000 (1%)]	Loss: 1.133197
Train Epoch: 5 [1280/50000 (3%)]	Loss: 1.255941
Train Epoch: 5 [1920/50000 (4%)]	Loss: 1.116746
Train Epoch: 5 [2560/50000 (5%)]	Loss: 1.236065
Train Epoch: 5 [3200/50000 (6%)]	Loss: 0.926574
Train Epoch: 5 [3840/50000 (8%)]	Loss: 1.183098
Train Epoch: 5 [4480/50000 (9%)]	Loss: 1.238905
Train Epoch: 5 [5120/50000 (10%)]	Loss: 1.359305
Train Epoch: 5 [5760/50000 (12%)]	Loss: 1.218621
Train Epoch: 5 [6400/50000 (13%)]	Loss: 1.144480
Train Epoch: 5 [7040/50000 (14%)]	Loss: 1.174589
Train Epoch: 5 [7680/50000 (15%)]	Loss: 1.032003
Train Epoch: 5 [8320/50000 (17%)]	Loss: 1.046212
Train Epoch: 5 [8960/50000 (18%)]	Loss: 1.227601
Train Epoch: 5 [9600/50000 (19%)]	Loss: 1.324029
Train Epoch: 5 [10240/50000 (20%)]	Loss: 1.262223
Train Epoch: 5 [10880/50000 (22%)]	Loss: 1.059315
Train Epoch: 5 [11520/50000 (23%)]	Loss: 1.061510
Train Epoch: 5 [12160/50000 (24%)]	Loss: 1.204941
Train Epoch: 5 [12800/50000 (26%)]	Loss: 1.125161
Train Epoch: 5 [13440/50000 (27%)]	Loss: 1.067031
Train Epoch: 5 [14080/50000 (28%)]	Loss: 1.274390
Train Epoch: 5 [14720/50000 (29%)]	Loss: 1.160022
Train Epoch: 5 [15360/50000 (31%)]	Loss: 1.119810
Train Epoch: 5 [16000/50000 (32%)]	Loss: 1.520325
Train Epoch: 5 [16640/50000 (33%)]	Loss: 1.226427
Train Epoch: 5 [17280/50000 (35%)]	Loss: 1.399533
Train Epoch: 5 [17920/50000 (36%)]	Loss: 1.305408
Train Epoch: 5 [18560/50000 (37%)]	Loss: 1.026983
Train Epoch: 5 [19200/50000 (38%)]	Loss: 1.224125
Train Epoch: 5 [19840/50000 (40%)]	Loss: 1.317264
Train Epoch: 5 [20480/50000 (41%)]	Loss: 1.096491
Train Epoch: 5 [21120/50000 (42%)]	Loss: 1.242739
Train Epoch: 5 [21760/50000 (43%)]	Loss: 1.091948
Train Epoch: 5 [22400/50000 (45%)]	Loss: 1.120600
Train Epoch: 5 [23040/50000 (46%)]	Loss: 1.224965
Train Epoch: 5 [23680/50000 (47%)]	Loss: 1.201894
Train Epoch: 5 [24320/50000 (49%)]	Loss: 1.239331
Train Epoch: 5 [24960/50000 (50%)]	Loss: 1.439321
Train Epoch: 5 [25600/50000 (51%)]	Loss: 0.959924
Train Epoch: 5 [26240/50000 (52%)]	Loss: 1.375550
Train Epoch: 5 [26880/50000 (54%)]	Loss: 1.096322
Train Epoch: 5 [27520/50000 (55%)]	Loss: 1.249927
Train Epoch: 5 [28160/50000 (56%)]	Loss: 1.321437
Train Epoch: 5 [28800/50000 (58%)]	Loss: 1.269089
Train Epoch: 5 [29440/50000 (59%)]	Loss: 1.219380
Train Epoch: 5 [30080/50000 (60%)]	Loss: 1.380494
Train Epoch: 5 [30720/50000 (61%)]	Loss: 1.089374
Train Epoch: 5 [31360/50000 (63%)]	Loss: 1.166920
Train Epoch: 5 [32000/50000 (64%)]	Loss: 1.503297
Train Epoch: 5 [32640/50000 (65%)]	Loss: 1.421054
Train Epoch: 5 [33280/50000 (66%)]	Loss: 1.218373
Train Epoch: 5 [33920/50000 (68%)]	Loss: 1.314114
Train Epoch: 5 [34560/50000 (69%)]	Loss: 1.323611
Train Epoch: 5 [35200/50000 (70%)]	Loss: 1.361807
Train Epoch: 5 [35840/50000 (72%)]	Loss: 1.197783
Train Epoch: 5 [36480/50000 (73%)]	Loss: 1.383361
Train Epoch: 5 [37120/50000 (74%)]	Loss: 1.252002
Train Epoch: 5 [37760/50000 (75%)]	Loss: 1.223987
Train Epoch: 5 [38400/50000 (77%)]	Loss: 1.174648
Train Epoch: 5 [39040/50000 (78%)]	Loss: 1.358514
Train Epoch: 5 [39680/50000 (79%)]	Loss: 1.122141
Train Epoch: 5 [40320/50000 (81%)]	Loss: 1.239308
Train Epoch: 5 [40960/50000 (82%)]	Loss: 1.256678
Train Epoch: 5 [41600/50000 (83%)]	Loss: 1.403788
Train Epoch: 5 [42240/50000 (84%)]	Loss: 0.945804
Train Epoch: 5 [42880/50000 (86%)]	Loss: 1.418401
Train Epoch: 5 [43520/50000 (87%)]	Loss: 1.242038
Train Epoch: 5 [44160/50000 (88%)]	Loss: 1.178189
Train Epoch: 5 [44800/50000 (90%)]	Loss: 1.132662
Train Epoch: 5 [45440/50000 (91%)]	Loss: 1.186559
Train Epoch: 5 [46080/50000 (92%)]	Loss: 1.053338
Train Epoch: 5 [46720/50000 (93%)]	Loss: 1.161613
Train Epoch: 5 [47360/50000 (95%)]	Loss: 0.992093
Train Epoch: 5 [48000/50000 (96%)]	Loss: 1.220029
Train Epoch: 5 [48640/50000 (97%)]	Loss: 0.987230
Train Epoch: 5 [49280/50000 (98%)]	Loss: 1.330422
Train Epoch: 5 [49920/50000 (100%)]	Loss: 1.337001

Test set: Average loss: 1.2406, Accuracy: 5564/10000 (56%)

Train Epoch: 6 [0/50000 (0%)]	Loss: 1.294107
Train Epoch: 6 [640/50000 (1%)]	Loss: 1.076030
Train Epoch: 6 [1280/50000 (3%)]	Loss: 1.143880
Train Epoch: 6 [1920/50000 (4%)]	Loss: 0.804564
Train Epoch: 6 [2560/50000 (5%)]	Loss: 1.198344
Train Epoch: 6 [3200/50000 (6%)]	Loss: 1.461010
Train Epoch: 6 [3840/50000 (8%)]	Loss: 1.005788
Train Epoch: 6 [4480/50000 (9%)]	Loss: 1.203327
Train Epoch: 6 [5120/50000 (10%)]	Loss: 1.146721
Train Epoch: 6 [5760/50000 (12%)]	Loss: 1.277295
Train Epoch: 6 [6400/50000 (13%)]	Loss: 1.167826
Train Epoch: 6 [7040/50000 (14%)]	Loss: 1.151743
Train Epoch: 6 [7680/50000 (15%)]	Loss: 1.351259
Train Epoch: 6 [8320/50000 (17%)]	Loss: 1.010704
Train Epoch: 6 [8960/50000 (18%)]	Loss: 0.990501
Train Epoch: 6 [9600/50000 (19%)]	Loss: 1.260333
Train Epoch: 6 [10240/50000 (20%)]	Loss: 0.977023
Train Epoch: 6 [10880/50000 (22%)]	Loss: 0.974628
Train Epoch: 6 [11520/50000 (23%)]	Loss: 1.130412
Train Epoch: 6 [12160/50000 (24%)]	Loss: 1.013300
Train Epoch: 6 [12800/50000 (26%)]	Loss: 1.288398
Train Epoch: 6 [13440/50000 (27%)]	Loss: 0.954830
Train Epoch: 6 [14080/50000 (28%)]	Loss: 1.141308
Train Epoch: 6 [14720/50000 (29%)]	Loss: 1.053300
Train Epoch: 6 [15360/50000 (31%)]	Loss: 1.385455
Train Epoch: 6 [16000/50000 (32%)]	Loss: 1.021367
Train Epoch: 6 [16640/50000 (33%)]	Loss: 1.337888
Train Epoch: 6 [17280/50000 (35%)]	Loss: 1.458092
Train Epoch: 6 [17920/50000 (36%)]	Loss: 0.961705
Train Epoch: 6 [18560/50000 (37%)]	Loss: 1.088531
Train Epoch: 6 [19200/50000 (38%)]	Loss: 1.006762
Train Epoch: 6 [19840/50000 (40%)]	Loss: 1.155338
Train Epoch: 6 [20480/50000 (41%)]	Loss: 1.186103
Train Epoch: 6 [21120/50000 (42%)]	Loss: 1.358949
Train Epoch: 6 [21760/50000 (43%)]	Loss: 1.061545
Train Epoch: 6 [22400/50000 (45%)]	Loss: 1.291943
Train Epoch: 6 [23040/50000 (46%)]	Loss: 1.286673
Train Epoch: 6 [23680/50000 (47%)]	Loss: 1.054450
Train Epoch: 6 [24320/50000 (49%)]	Loss: 1.279654
Train Epoch: 6 [24960/50000 (50%)]	Loss: 1.091016
Train Epoch: 6 [25600/50000 (51%)]	Loss: 1.114034
Train Epoch: 6 [26240/50000 (52%)]	Loss: 1.074285
Train Epoch: 6 [26880/50000 (54%)]	Loss: 1.207639
Train Epoch: 6 [27520/50000 (55%)]	Loss: 1.035142
Train Epoch: 6 [28160/50000 (56%)]	Loss: 1.097257
Train Epoch: 6 [28800/50000 (58%)]	Loss: 1.153214
Train Epoch: 6 [29440/50000 (59%)]	Loss: 1.268563
Train Epoch: 6 [30080/50000 (60%)]	Loss: 1.142697
Train Epoch: 6 [30720/50000 (61%)]	Loss: 0.969169
Train Epoch: 6 [31360/50000 (63%)]	Loss: 1.086179
Train Epoch: 6 [32000/50000 (64%)]	Loss: 1.206302
Train Epoch: 6 [32640/50000 (65%)]	Loss: 1.310626
Train Epoch: 6 [33280/50000 (66%)]	Loss: 1.205313
Train Epoch: 6 [33920/50000 (68%)]	Loss: 1.316682
Train Epoch: 6 [34560/50000 (69%)]	Loss: 1.399464
Train Epoch: 6 [35200/50000 (70%)]	Loss: 0.935241
Train Epoch: 6 [35840/50000 (72%)]	Loss: 1.208363
Train Epoch: 6 [36480/50000 (73%)]	Loss: 1.319098
Train Epoch: 6 [37120/50000 (74%)]	Loss: 1.014982
Train Epoch: 6 [37760/50000 (75%)]	Loss: 1.075221
Train Epoch: 6 [38400/50000 (77%)]	Loss: 1.322136
Train Epoch: 6 [39040/50000 (78%)]	Loss: 1.132068
Train Epoch: 6 [39680/50000 (79%)]	Loss: 0.940457
Train Epoch: 6 [40320/50000 (81%)]	Loss: 0.926928
Train Epoch: 6 [40960/50000 (82%)]	Loss: 1.053174
Train Epoch: 6 [41600/50000 (83%)]	Loss: 1.167490
Train Epoch: 6 [42240/50000 (84%)]	Loss: 1.055591
Train Epoch: 6 [42880/50000 (86%)]	Loss: 1.053160
Train Epoch: 6 [43520/50000 (87%)]	Loss: 1.113515
Train Epoch: 6 [44160/50000 (88%)]	Loss: 0.910549
Train Epoch: 6 [44800/50000 (90%)]	Loss: 1.234298
Train Epoch: 6 [45440/50000 (91%)]	Loss: 1.024721
Train Epoch: 6 [46080/50000 (92%)]	Loss: 1.229620
Train Epoch: 6 [46720/50000 (93%)]	Loss: 0.968234
Train Epoch: 6 [47360/50000 (95%)]	Loss: 1.078589
Train Epoch: 6 [48000/50000 (96%)]	Loss: 1.038514
Train Epoch: 6 [48640/50000 (97%)]	Loss: 1.456464
Train Epoch: 6 [49280/50000 (98%)]	Loss: 0.867064
Train Epoch: 6 [49920/50000 (100%)]	Loss: 1.096560

Test set: Average loss: 1.2323, Accuracy: 5640/10000 (56%)

Train Epoch: 7 [0/50000 (0%)]	Loss: 1.259657
Train Epoch: 7 [640/50000 (1%)]	Loss: 1.142298
Train Epoch: 7 [1280/50000 (3%)]	Loss: 1.197640
Train Epoch: 7 [1920/50000 (4%)]	Loss: 1.094428
Train Epoch: 7 [2560/50000 (5%)]	Loss: 1.212835
Train Epoch: 7 [3200/50000 (6%)]	Loss: 1.171647
Train Epoch: 7 [3840/50000 (8%)]	Loss: 1.178333
Train Epoch: 7 [4480/50000 (9%)]	Loss: 1.151591
Train Epoch: 7 [5120/50000 (10%)]	Loss: 0.804631
Train Epoch: 7 [5760/50000 (12%)]	Loss: 1.033653
Train Epoch: 7 [6400/50000 (13%)]	Loss: 1.294702
Train Epoch: 7 [7040/50000 (14%)]	Loss: 0.953857
Train Epoch: 7 [7680/50000 (15%)]	Loss: 1.001062
Train Epoch: 7 [8320/50000 (17%)]	Loss: 1.254230
Train Epoch: 7 [8960/50000 (18%)]	Loss: 1.189610
Train Epoch: 7 [9600/50000 (19%)]	Loss: 1.099068
Train Epoch: 7 [10240/50000 (20%)]	Loss: 0.919710
Train Epoch: 7 [10880/50000 (22%)]	Loss: 1.123262
Train Epoch: 7 [11520/50000 (23%)]	Loss: 1.012293
Train Epoch: 7 [12160/50000 (24%)]	Loss: 1.008054
Train Epoch: 7 [12800/50000 (26%)]	Loss: 1.157341
Train Epoch: 7 [13440/50000 (27%)]	Loss: 1.145163
Train Epoch: 7 [14080/50000 (28%)]	Loss: 1.035322
Train Epoch: 7 [14720/50000 (29%)]	Loss: 1.038333
Train Epoch: 7 [15360/50000 (31%)]	Loss: 1.159069
Train Epoch: 7 [16000/50000 (32%)]	Loss: 1.190673
Train Epoch: 7 [16640/50000 (33%)]	Loss: 0.912957
Train Epoch: 7 [17280/50000 (35%)]	Loss: 1.438032
Train Epoch: 7 [17920/50000 (36%)]	Loss: 1.102762
Train Epoch: 7 [18560/50000 (37%)]	Loss: 1.033348
Train Epoch: 7 [19200/50000 (38%)]	Loss: 0.972180
Train Epoch: 7 [19840/50000 (40%)]	Loss: 1.177395
Train Epoch: 7 [20480/50000 (41%)]	Loss: 1.272233
Train Epoch: 7 [21120/50000 (42%)]	Loss: 1.106514
Train Epoch: 7 [21760/50000 (43%)]	Loss: 1.257112
Train Epoch: 7 [22400/50000 (45%)]	Loss: 0.936901
Train Epoch: 7 [23040/50000 (46%)]	Loss: 0.970605
Train Epoch: 7 [23680/50000 (47%)]	Loss: 0.931822
Train Epoch: 7 [24320/50000 (49%)]	Loss: 0.922658
Train Epoch: 7 [24960/50000 (50%)]	Loss: 1.045256
Train Epoch: 7 [25600/50000 (51%)]	Loss: 0.919892
Train Epoch: 7 [26240/50000 (52%)]	Loss: 1.135273
Train Epoch: 7 [26880/50000 (54%)]	Loss: 1.220780
Train Epoch: 7 [27520/50000 (55%)]	Loss: 1.051692
Train Epoch: 7 [28160/50000 (56%)]	Loss: 1.164889
Train Epoch: 7 [28800/50000 (58%)]	Loss: 1.085477
Train Epoch: 7 [29440/50000 (59%)]	Loss: 1.076715
Train Epoch: 7 [30080/50000 (60%)]	Loss: 0.987024
Train Epoch: 7 [30720/50000 (61%)]	Loss: 1.097356
Train Epoch: 7 [31360/50000 (63%)]	Loss: 1.451029
Train Epoch: 7 [32000/50000 (64%)]	Loss: 1.199328
Train Epoch: 7 [32640/50000 (65%)]	Loss: 1.048133
Train Epoch: 7 [33280/50000 (66%)]	Loss: 0.973138
Train Epoch: 7 [33920/50000 (68%)]	Loss: 1.021813
Train Epoch: 7 [34560/50000 (69%)]	Loss: 1.267141
Train Epoch: 7 [35200/50000 (70%)]	Loss: 1.202436
Train Epoch: 7 [35840/50000 (72%)]	Loss: 1.195908
Train Epoch: 7 [36480/50000 (73%)]	Loss: 0.951825
Train Epoch: 7 [37120/50000 (74%)]	Loss: 1.169059
Train Epoch: 7 [37760/50000 (75%)]	Loss: 1.101759
Train Epoch: 7 [38400/50000 (77%)]	Loss: 1.267733
Train Epoch: 7 [39040/50000 (78%)]	Loss: 1.191783
Train Epoch: 7 [39680/50000 (79%)]	Loss: 1.248074
Train Epoch: 7 [40320/50000 (81%)]	Loss: 0.911359
Train Epoch: 7 [40960/50000 (82%)]	Loss: 1.145764
Train Epoch: 7 [41600/50000 (83%)]	Loss: 1.212530
Train Epoch: 7 [42240/50000 (84%)]	Loss: 1.155478
Train Epoch: 7 [42880/50000 (86%)]	Loss: 1.248468
Train Epoch: 7 [43520/50000 (87%)]	Loss: 1.207988
Train Epoch: 7 [44160/50000 (88%)]	Loss: 1.640075
Train Epoch: 7 [44800/50000 (90%)]	Loss: 1.298544
Train Epoch: 7 [45440/50000 (91%)]	Loss: 1.154577
Train Epoch: 7 [46080/50000 (92%)]	Loss: 1.115201
Train Epoch: 7 [46720/50000 (93%)]	Loss: 1.198518
Train Epoch: 7 [47360/50000 (95%)]	Loss: 1.244714
Train Epoch: 7 [48000/50000 (96%)]	Loss: 1.226714
Train Epoch: 7 [48640/50000 (97%)]	Loss: 1.032409
Train Epoch: 7 [49280/50000 (98%)]	Loss: 1.173413
Train Epoch: 7 [49920/50000 (100%)]	Loss: 1.145262

Test set: Average loss: 1.2744, Accuracy: 5535/10000 (55%)

Train Epoch: 8 [0/50000 (0%)]	Loss: 1.317575
Train Epoch: 8 [640/50000 (1%)]	Loss: 1.091076
Train Epoch: 8 [1280/50000 (3%)]	Loss: 0.829476
Train Epoch: 8 [1920/50000 (4%)]	Loss: 1.032486
Train Epoch: 8 [2560/50000 (5%)]	Loss: 1.266073
Train Epoch: 8 [3200/50000 (6%)]	Loss: 1.135764
Train Epoch: 8 [3840/50000 (8%)]	Loss: 1.107314
Train Epoch: 8 [4480/50000 (9%)]	Loss: 1.174266
Train Epoch: 8 [5120/50000 (10%)]	Loss: 1.018157
Train Epoch: 8 [5760/50000 (12%)]	Loss: 1.215114
Train Epoch: 8 [6400/50000 (13%)]	Loss: 1.069255
Train Epoch: 8 [7040/50000 (14%)]	Loss: 1.129509
Train Epoch: 8 [7680/50000 (15%)]	Loss: 1.281269
Train Epoch: 8 [8320/50000 (17%)]	Loss: 1.226652
Train Epoch: 8 [8960/50000 (18%)]	Loss: 1.126550
Train Epoch: 8 [9600/50000 (19%)]	Loss: 1.210916
Train Epoch: 8 [10240/50000 (20%)]	Loss: 1.000816
Train Epoch: 8 [10880/50000 (22%)]	Loss: 1.188282
Train Epoch: 8 [11520/50000 (23%)]	Loss: 1.069835
Train Epoch: 8 [12160/50000 (24%)]	Loss: 0.953737
Train Epoch: 8 [12800/50000 (26%)]	Loss: 1.239916
Train Epoch: 8 [13440/50000 (27%)]	Loss: 1.419442
Train Epoch: 8 [14080/50000 (28%)]	Loss: 1.065208
Train Epoch: 8 [14720/50000 (29%)]	Loss: 1.385827
Train Epoch: 8 [15360/50000 (31%)]	Loss: 1.257588
Train Epoch: 8 [16000/50000 (32%)]	Loss: 1.228138
Train Epoch: 8 [16640/50000 (33%)]	Loss: 1.140245
Train Epoch: 8 [17280/50000 (35%)]	Loss: 1.228857
Train Epoch: 8 [17920/50000 (36%)]	Loss: 1.034999
Train Epoch: 8 [18560/50000 (37%)]	Loss: 1.070533
Train Epoch: 8 [19200/50000 (38%)]	Loss: 1.158831
Train Epoch: 8 [19840/50000 (40%)]	Loss: 0.931073
Train Epoch: 8 [20480/50000 (41%)]	Loss: 0.879403
Train Epoch: 8 [21120/50000 (42%)]	Loss: 0.933083
Train Epoch: 8 [21760/50000 (43%)]	Loss: 0.943778
Train Epoch: 8 [22400/50000 (45%)]	Loss: 1.248006
Train Epoch: 8 [23040/50000 (46%)]	Loss: 1.127589
Train Epoch: 8 [23680/50000 (47%)]	Loss: 0.936665
Train Epoch: 8 [24320/50000 (49%)]	Loss: 0.936823
Train Epoch: 8 [24960/50000 (50%)]	Loss: 1.045675
Train Epoch: 8 [25600/50000 (51%)]	Loss: 0.980460
Train Epoch: 8 [26240/50000 (52%)]	Loss: 1.245111
Train Epoch: 8 [26880/50000 (54%)]	Loss: 1.083245
Train Epoch: 8 [27520/50000 (55%)]	Loss: 0.956864
Train Epoch: 8 [28160/50000 (56%)]	Loss: 1.322951
Train Epoch: 8 [28800/50000 (58%)]	Loss: 1.119469
Train Epoch: 8 [29440/50000 (59%)]	Loss: 0.986755
Train Epoch: 8 [30080/50000 (60%)]	Loss: 1.498300
Train Epoch: 8 [30720/50000 (61%)]	Loss: 0.896741
Train Epoch: 8 [31360/50000 (63%)]	Loss: 1.330052
Train Epoch: 8 [32000/50000 (64%)]	Loss: 1.177285
Train Epoch: 8 [32640/50000 (65%)]	Loss: 0.933658
Train Epoch: 8 [33280/50000 (66%)]	Loss: 1.129594
Train Epoch: 8 [33920/50000 (68%)]	Loss: 1.213060
Train Epoch: 8 [34560/50000 (69%)]	Loss: 0.963461
Train Epoch: 8 [35200/50000 (70%)]	Loss: 1.098998
Train Epoch: 8 [35840/50000 (72%)]	Loss: 1.019652
Train Epoch: 8 [36480/50000 (73%)]	Loss: 0.976270
Train Epoch: 8 [37120/50000 (74%)]	Loss: 1.010774
Train Epoch: 8 [37760/50000 (75%)]	Loss: 1.069298
Train Epoch: 8 [38400/50000 (77%)]	Loss: 0.909558
Train Epoch: 8 [39040/50000 (78%)]	Loss: 1.142531
Train Epoch: 8 [39680/50000 (79%)]	Loss: 1.108397
Train Epoch: 8 [40320/50000 (81%)]	Loss: 0.977082
Train Epoch: 8 [40960/50000 (82%)]	Loss: 1.089954
Train Epoch: 8 [41600/50000 (83%)]	Loss: 1.197649
Train Epoch: 8 [42240/50000 (84%)]	Loss: 1.197317
Train Epoch: 8 [42880/50000 (86%)]	Loss: 1.013878
Train Epoch: 8 [43520/50000 (87%)]	Loss: 0.978553
Train Epoch: 8 [44160/50000 (88%)]	Loss: 0.934539
Train Epoch: 8 [44800/50000 (90%)]	Loss: 0.976169
Train Epoch: 8 [45440/50000 (91%)]	Loss: 1.009297
Train Epoch: 8 [46080/50000 (92%)]	Loss: 1.046346
Train Epoch: 8 [46720/50000 (93%)]	Loss: 0.929104
Train Epoch: 8 [47360/50000 (95%)]	Loss: 1.180241
Train Epoch: 8 [48000/50000 (96%)]	Loss: 1.178066
Train Epoch: 8 [48640/50000 (97%)]	Loss: 1.110379
Train Epoch: 8 [49280/50000 (98%)]	Loss: 1.346457
Train Epoch: 8 [49920/50000 (100%)]	Loss: 1.210698

Test set: Average loss: 1.1761, Accuracy: 5897/10000 (59%)

Train Epoch: 9 [0/50000 (0%)]	Loss: 1.297561
Train Epoch: 9 [640/50000 (1%)]	Loss: 1.242794
Train Epoch: 9 [1280/50000 (3%)]	Loss: 1.189888
Train Epoch: 9 [1920/50000 (4%)]	Loss: 1.189721
Train Epoch: 9 [2560/50000 (5%)]	Loss: 0.928716
Train Epoch: 9 [3200/50000 (6%)]	Loss: 1.006349
Train Epoch: 9 [3840/50000 (8%)]	Loss: 1.303151
Train Epoch: 9 [4480/50000 (9%)]	Loss: 1.170540
Train Epoch: 9 [5120/50000 (10%)]	Loss: 1.157369
Train Epoch: 9 [5760/50000 (12%)]	Loss: 1.069863
Train Epoch: 9 [6400/50000 (13%)]	Loss: 1.080377
Train Epoch: 9 [7040/50000 (14%)]	Loss: 0.980811
Train Epoch: 9 [7680/50000 (15%)]	Loss: 1.013740
Train Epoch: 9 [8320/50000 (17%)]	Loss: 1.061553
Train Epoch: 9 [8960/50000 (18%)]	Loss: 0.957188
Train Epoch: 9 [9600/50000 (19%)]	Loss: 0.971652
Train Epoch: 9 [10240/50000 (20%)]	Loss: 0.985674
Train Epoch: 9 [10880/50000 (22%)]	Loss: 1.193167
Train Epoch: 9 [11520/50000 (23%)]	Loss: 1.273659
Train Epoch: 9 [12160/50000 (24%)]	Loss: 0.999477
Train Epoch: 9 [12800/50000 (26%)]	Loss: 0.766888
Train Epoch: 9 [13440/50000 (27%)]	Loss: 1.228657
Train Epoch: 9 [14080/50000 (28%)]	Loss: 1.191001
Train Epoch: 9 [14720/50000 (29%)]	Loss: 0.949604
Train Epoch: 9 [15360/50000 (31%)]	Loss: 1.396086
Train Epoch: 9 [16000/50000 (32%)]	Loss: 1.105575
Train Epoch: 9 [16640/50000 (33%)]	Loss: 1.375776
Train Epoch: 9 [17280/50000 (35%)]	Loss: 1.018983
Train Epoch: 9 [17920/50000 (36%)]	Loss: 1.109476
Train Epoch: 9 [18560/50000 (37%)]	Loss: 1.050700
Train Epoch: 9 [19200/50000 (38%)]	Loss: 0.950169
Train Epoch: 9 [19840/50000 (40%)]	Loss: 0.941315
Train Epoch: 9 [20480/50000 (41%)]	Loss: 0.875120
Train Epoch: 9 [21120/50000 (42%)]	Loss: 1.502672
Train Epoch: 9 [21760/50000 (43%)]	Loss: 1.055388
Train Epoch: 9 [22400/50000 (45%)]	Loss: 1.018230
Train Epoch: 9 [23040/50000 (46%)]	Loss: 1.455761
Train Epoch: 9 [23680/50000 (47%)]	Loss: 0.983607
Train Epoch: 9 [24320/50000 (49%)]	Loss: 1.125270
Train Epoch: 9 [24960/50000 (50%)]	Loss: 0.879530
Train Epoch: 9 [25600/50000 (51%)]	Loss: 1.012308
Train Epoch: 9 [26240/50000 (52%)]	Loss: 0.745379
Train Epoch: 9 [26880/50000 (54%)]	Loss: 1.034528
Train Epoch: 9 [27520/50000 (55%)]	Loss: 1.033346
Train Epoch: 9 [28160/50000 (56%)]	Loss: 1.038836
Train Epoch: 9 [28800/50000 (58%)]	Loss: 1.172081
Train Epoch: 9 [29440/50000 (59%)]	Loss: 0.944392
Train Epoch: 9 [30080/50000 (60%)]	Loss: 0.924668
Train Epoch: 9 [30720/50000 (61%)]	Loss: 0.951646
Train Epoch: 9 [31360/50000 (63%)]	Loss: 0.881092
Train Epoch: 9 [32000/50000 (64%)]	Loss: 0.992526
Train Epoch: 9 [32640/50000 (65%)]	Loss: 1.097238
Train Epoch: 9 [33280/50000 (66%)]	Loss: 1.048317
Train Epoch: 9 [33920/50000 (68%)]	Loss: 1.048791
Train Epoch: 9 [34560/50000 (69%)]	Loss: 1.060812
Train Epoch: 9 [35200/50000 (70%)]	Loss: 1.277424
Train Epoch: 9 [35840/50000 (72%)]	Loss: 1.424361
Train Epoch: 9 [36480/50000 (73%)]	Loss: 1.025828
Train Epoch: 9 [37120/50000 (74%)]	Loss: 1.092653
Train Epoch: 9 [37760/50000 (75%)]	Loss: 0.772143
Train Epoch: 9 [38400/50000 (77%)]	Loss: 1.161311
Train Epoch: 9 [39040/50000 (78%)]	Loss: 0.996327
Train Epoch: 9 [39680/50000 (79%)]	Loss: 1.167324
Train Epoch: 9 [40320/50000 (81%)]	Loss: 1.407117
Train Epoch: 9 [40960/50000 (82%)]	Loss: 1.121193
Train Epoch: 9 [41600/50000 (83%)]	Loss: 0.928374
Train Epoch: 9 [42240/50000 (84%)]	Loss: 0.950045
Train Epoch: 9 [42880/50000 (86%)]	Loss: 1.231665
Train Epoch: 9 [43520/50000 (87%)]	Loss: 1.089976
Train Epoch: 9 [44160/50000 (88%)]	Loss: 1.167236
Train Epoch: 9 [44800/50000 (90%)]	Loss: 0.887712
Train Epoch: 9 [45440/50000 (91%)]	Loss: 0.933750
Train Epoch: 9 [46080/50000 (92%)]	Loss: 1.195859
Train Epoch: 9 [46720/50000 (93%)]	Loss: 1.120610
Train Epoch: 9 [47360/50000 (95%)]	Loss: 1.067099
Train Epoch: 9 [48000/50000 (96%)]	Loss: 0.840605
Train Epoch: 9 [48640/50000 (97%)]	Loss: 0.904040
Train Epoch: 9 [49280/50000 (98%)]	Loss: 1.280596
Train Epoch: 9 [49920/50000 (100%)]	Loss: 1.046842

Test set: Average loss: 1.1206, Accuracy: 6068/10000 (61%)
"""
