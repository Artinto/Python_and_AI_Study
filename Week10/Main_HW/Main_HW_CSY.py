# 파이썬 버전이 어떤 것이든 파이썬 3 문법인 print()를 통해 출력 가능
# touch 모듈 내에 있는 nn, optim, cuda(GPU 사용)를 불러옴
# torch.utils.data는 SGD(Stochastic Gradient Descent)의 반복 연산을 실행할 때 사용하는 미니 배치용 유틸리티 함수 포함
# torchvision 유명 데이터셋, 구현되어 있는 모델, 이미지 전처리 도구를 포함하고 있는 패키지로 datasets, transforms(전처리 방법들)을 불러옴
#  torch.nn은 클래스로 정의됨, torch.nn.functional은 함수로 정의됨
# time 모듈 불러옴
from __future__ import print_function
from torch import nn, optim, cuda
from torch.utils import data
from torchvision import datasets, transforms
import torch.nn.functional as F

# Training settings
# 배치 사이즈는 50로 설정
# cuda(GPU) 사용 가능하다면 사용, 안되면 CPU 사용
# 사용 device 출력
batch_size = 50
device = 'cuda' if cuda.is_available() else 'cpu'
print(f'Training CIFAR10 Model on {device}\n{"=" * 44}')

# CIFAR10 Dataset
# 훈련할 데이터셋을 불러옴/train을 True로 하면 훈련 데이터 셋을 리턴받음/transform을 통해 현재 데이터를 파이토치 텐서로 변환/download는 해당 경로에 CIFAR10 데이터가 없으면 다운로드함
train_dataset = datasets.CIFAR10(root='./CIFAR10_data/', train=True, transform=transforms.ToTensor(), download=True)

# 테스트할 데이터셋을 불러옴/train을 False로 하면 테스트 데이터 셋을 리턴받음/transform을 통해 현재 데이터를 파이토치 텐서로 변환
test_dataset = datasets.CIFAR10(root='./CIFAR10_data/', train=False, transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
# 훈련할 데이터 로더/훈련 데이터셋/배치 사이즈 64/무작위 순서
train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# 테스트할 데이터 로더/테스트 데이터셋/배치 사이즈 64/순서는 차례대로
test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# torch.nn.Module을 상속받는 파이썬 클래스 정의
# 모델의 생성자 정의 (객체가 갖는 속성값을 초기화하는 역할, 객체가 생성될 때 자동으로 호출됨)
class Net(nn.Module):

    def __init__(self):
        # super() 함수를 통해 nn.Module의 생성자를 호출함
        # nn.Linear() 함수를 이용하여 8개 층의 모델을 만듦((입력,출력)32*32*3(3072)와 10은 고정)
        super(Net, self).__init__()
        self.l1 = nn.Linear(3072, 2000)
        self.l2 = nn.Linear(2000, 1500)
        self.l3 = nn.Linear(1500, 800)
        self.l4 = nn.Linear(800, 400)
        self.l5 = nn.Linear(400, 200)
        self.l6 = nn.Linear(200, 10)

    # 모델 객체와 학습 데이터인 x를 받아 forward 연산하는 함수로 Net(입력 데이터) 형식으로 객체를 호출하면 자동으로 forward 연산 수행됨
    def forward(self, x):
        # view 함수는 원소의 수를 유지하면서 텐서의 크기를 변경함/3 x 32 x 32 벡터(32 x 32 크기, 3(R,G,B)가지 색)를 3072 길이만큼으로 변경/-1은 첫번째 차원은 파이토치에 맡겨 설정한다는 의미/3072는 두번째 차원의 길이를 의미
        # x를 선형함수에 넣고 활성화 함수인 relu함수를 통해 나온 결과값을 x에 저장
        # 마지막 return 값은 활성화 함수 사용하지 않음(logits을 사용하기 때문)
        x = x.view(-1, 3072)  # Flatten the data (n, 3, 32, 32)-> (n, 3072)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = F.relu(self.l5(x))
        return self.l6(x)

# Net 클래스 변수 model 선언
# device에 모델 등록
# 손실함수 정의 코드로, CrossEntropyLoss 함수 사용
# SGD(확률적 경사 하강법)는 경사 하강법의 일종이고 model.parameters()를 이용하여 parameter를 전달함. lr은 학습률이며 momentum은 SGD에 관성을 더해줌(이전 이동 값을 고려하여 일정 비율만큼 다음 값을 결정)
model = Net()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.06, momentum=0.5)

# 훈련 과정
def train(epoch):
    # 학습 모드로 전환
    model.train()
    # train_loader로 학습
    # 각 data와 target을 전에 설정했던 device에 보냄
    # 미분을 통해 얻은 기울기가 이전에 계산된 기울기 값에 누적되기 때문에 기울기 값을 0으로 초기화 함
    # data를 model에 넣어 예측값 output을 도출
    # 손실값을 criterion함수를 이용해 도출 (예측값, 결과값)
    # 역전파 실행, 손실 함수를 미분하여 기울기 계산
    # step() 함수를 호출하여 parameter를 업데이트함
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        # 배치 사이즈가 10의 배수일때마다 학습을 반복한 수, Batch Status, loss값 출력
        if batch_idx % 10 == 0:
            print('Train Epoch: {} | Batch Status: {}/{} ({:.0f}%) | Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

# 테스트 과정
def test():
    # 평가 모드로 전환
    # 정확도 계산을 위한 변수 설정
    model.eval()
    test_loss = 0
    correct = 0
    # test_loader로 모델 시험
    # 각 data와 target을 전에 설정했던 device에 보냄
    # data를 model에 넣어 예측값 output을 도출
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        # sum up batch loss
        # 배치 손실값의 합
        test_loss += criterion(output, target).item()
        # get the index of the max
        # 최대값의 index를 반환/keepdim은 벡터 차원을 유지 시킬건지 아닌지를 설정
        # pred값과 target값을 비교하여 일치하는지 검사한 후 일치하는 것들의 개수의 합을 저장
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    # 테스트 손실값을 test_loader.dataset길이로 나눔
    # 테스트에 대한 손실값과 정확도를 계산
    test_loss /= len(test_loader.dataset)
    print(f'===========================\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} 'f'({100. * correct / len(test_loader.dataset):.0f}%)')

# main에서 실행
# time 함수로 현재 시간 구함
# 15번 반복
if __name__ == '__main__':
    for epoch in range(1, 15):
        # 학습 훈련을 시작
        # 테스트를 시작
        train(epoch)
        test()
      
'''
Training CIFAR10 Model on cpu
============================================
Files already downloaded and verified
Train Epoch: 1 | Batch Status: 0/50000 (0%) | Loss: 2.305635
Train Epoch: 1 | Batch Status: 500/50000 (1%) | Loss: 2.309302
Train Epoch: 1 | Batch Status: 1000/50000 (2%) | Loss: 2.300823
Train Epoch: 1 | Batch Status: 1500/50000 (3%) | Loss: 2.292138
Train Epoch: 1 | Batch Status: 2000/50000 (4%) | Loss: 2.301250
Train Epoch: 1 | Batch Status: 2500/50000 (5%) | Loss: 2.296802
Train Epoch: 1 | Batch Status: 3000/50000 (6%) | Loss: 2.294841
Train Epoch: 1 | Batch Status: 3500/50000 (7%) | Loss: 2.304297
Train Epoch: 1 | Batch Status: 4000/50000 (8%) | Loss: 2.313159
Train Epoch: 1 | Batch Status: 4500/50000 (9%) | Loss: 2.309890
Train Epoch: 1 | Batch Status: 5000/50000 (10%) | Loss: 2.290483
Train Epoch: 1 | Batch Status: 5500/50000 (11%) | Loss: 2.281312
Train Epoch: 1 | Batch Status: 6000/50000 (12%) | Loss: 2.296458
Train Epoch: 1 | Batch Status: 6500/50000 (13%) | Loss: 2.258031
Train Epoch: 1 | Batch Status: 7000/50000 (14%) | Loss: 2.223064
Train Epoch: 1 | Batch Status: 7500/50000 (15%) | Loss: 2.226757
Train Epoch: 1 | Batch Status: 8000/50000 (16%) | Loss: 2.149382
Train Epoch: 1 | Batch Status: 8500/50000 (17%) | Loss: 2.194294
Train Epoch: 1 | Batch Status: 9000/50000 (18%) | Loss: 2.209892
Train Epoch: 1 | Batch Status: 9500/50000 (19%) | Loss: 2.164560
Train Epoch: 1 | Batch Status: 10000/50000 (20%) | Loss: 1.984818
Train Epoch: 1 | Batch Status: 10500/50000 (21%) | Loss: 2.035470
Train Epoch: 1 | Batch Status: 11000/50000 (22%) | Loss: 2.162655
Train Epoch: 1 | Batch Status: 11500/50000 (23%) | Loss: 2.012411
Train Epoch: 1 | Batch Status: 12000/50000 (24%) | Loss: 2.092755
Train Epoch: 1 | Batch Status: 12500/50000 (25%) | Loss: 2.011216
Train Epoch: 1 | Batch Status: 13000/50000 (26%) | Loss: 2.009662
Train Epoch: 1 | Batch Status: 13500/50000 (27%) | Loss: 2.009751
Train Epoch: 1 | Batch Status: 14000/50000 (28%) | Loss: 2.056495
Train Epoch: 1 | Batch Status: 14500/50000 (29%) | Loss: 2.056727
Train Epoch: 1 | Batch Status: 15000/50000 (30%) | Loss: 2.128230
Train Epoch: 1 | Batch Status: 15500/50000 (31%) | Loss: 2.180259
Train Epoch: 1 | Batch Status: 16000/50000 (32%) | Loss: 2.108092
Train Epoch: 1 | Batch Status: 16500/50000 (33%) | Loss: 1.955430
Train Epoch: 1 | Batch Status: 17000/50000 (34%) | Loss: 2.063222
Train Epoch: 1 | Batch Status: 17500/50000 (35%) | Loss: 1.992221
Train Epoch: 1 | Batch Status: 18000/50000 (36%) | Loss: 2.007828
Train Epoch: 1 | Batch Status: 18500/50000 (37%) | Loss: 1.927778
Train Epoch: 1 | Batch Status: 19000/50000 (38%) | Loss: 1.965056
Train Epoch: 1 | Batch Status: 19500/50000 (39%) | Loss: 2.078329
Train Epoch: 1 | Batch Status: 20000/50000 (40%) | Loss: 2.209693
Train Epoch: 1 | Batch Status: 20500/50000 (41%) | Loss: 2.065916
Train Epoch: 1 | Batch Status: 21000/50000 (42%) | Loss: 1.784449
Train Epoch: 1 | Batch Status: 21500/50000 (43%) | Loss: 1.890628
Train Epoch: 1 | Batch Status: 22000/50000 (44%) | Loss: 2.070513
Train Epoch: 1 | Batch Status: 22500/50000 (45%) | Loss: 2.093381
Train Epoch: 1 | Batch Status: 23000/50000 (46%) | Loss: 1.948118
Train Epoch: 1 | Batch Status: 23500/50000 (47%) | Loss: 2.005945
Train Epoch: 1 | Batch Status: 24000/50000 (48%) | Loss: 2.161572
Train Epoch: 1 | Batch Status: 24500/50000 (49%) | Loss: 1.925899
Train Epoch: 1 | Batch Status: 25000/50000 (50%) | Loss: 1.928121
Train Epoch: 1 | Batch Status: 25500/50000 (51%) | Loss: 1.933433
Train Epoch: 1 | Batch Status: 26000/50000 (52%) | Loss: 2.084329
Train Epoch: 1 | Batch Status: 26500/50000 (53%) | Loss: 2.062515
Train Epoch: 1 | Batch Status: 27000/50000 (54%) | Loss: 1.911719
Train Epoch: 1 | Batch Status: 27500/50000 (55%) | Loss: 2.078615
Train Epoch: 1 | Batch Status: 28000/50000 (56%) | Loss: 1.760878
Train Epoch: 1 | Batch Status: 28500/50000 (57%) | Loss: 1.891728
Train Epoch: 1 | Batch Status: 29000/50000 (58%) | Loss: 2.024361
Train Epoch: 1 | Batch Status: 29500/50000 (59%) | Loss: 2.076386
Train Epoch: 1 | Batch Status: 30000/50000 (60%) | Loss: 1.883293
Train Epoch: 1 | Batch Status: 30500/50000 (61%) | Loss: 1.895501
Train Epoch: 1 | Batch Status: 31000/50000 (62%) | Loss: 1.961930
Train Epoch: 1 | Batch Status: 31500/50000 (63%) | Loss: 1.804492
Train Epoch: 1 | Batch Status: 32000/50000 (64%) | Loss: 2.051201
Train Epoch: 1 | Batch Status: 32500/50000 (65%) | Loss: 1.839195
Train Epoch: 1 | Batch Status: 33000/50000 (66%) | Loss: 1.871594
Train Epoch: 1 | Batch Status: 33500/50000 (67%) | Loss: 1.736735
Train Epoch: 1 | Batch Status: 34000/50000 (68%) | Loss: 2.031618
Train Epoch: 1 | Batch Status: 34500/50000 (69%) | Loss: 1.870960
Train Epoch: 1 | Batch Status: 35000/50000 (70%) | Loss: 1.889253
Train Epoch: 1 | Batch Status: 35500/50000 (71%) | Loss: 1.857296
Train Epoch: 1 | Batch Status: 36000/50000 (72%) | Loss: 2.030449
Train Epoch: 1 | Batch Status: 36500/50000 (73%) | Loss: 2.007931
Train Epoch: 1 | Batch Status: 37000/50000 (74%) | Loss: 1.780359
Train Epoch: 1 | Batch Status: 37500/50000 (75%) | Loss: 1.909130
Train Epoch: 1 | Batch Status: 38000/50000 (76%) | Loss: 1.943291
Train Epoch: 1 | Batch Status: 38500/50000 (77%) | Loss: 1.748819
Train Epoch: 1 | Batch Status: 39000/50000 (78%) | Loss: 1.849525
Train Epoch: 1 | Batch Status: 39500/50000 (79%) | Loss: 1.898938
Train Epoch: 1 | Batch Status: 40000/50000 (80%) | Loss: 1.877278
Train Epoch: 1 | Batch Status: 40500/50000 (81%) | Loss: 1.886957
Train Epoch: 1 | Batch Status: 41000/50000 (82%) | Loss: 1.923414
Train Epoch: 1 | Batch Status: 41500/50000 (83%) | Loss: 1.823644
Train Epoch: 1 | Batch Status: 42000/50000 (84%) | Loss: 1.961337
Train Epoch: 1 | Batch Status: 42500/50000 (85%) | Loss: 1.854052
Train Epoch: 1 | Batch Status: 43000/50000 (86%) | Loss: 2.172250
Train Epoch: 1 | Batch Status: 43500/50000 (87%) | Loss: 1.795434
Train Epoch: 1 | Batch Status: 44000/50000 (88%) | Loss: 1.999495
Train Epoch: 1 | Batch Status: 44500/50000 (89%) | Loss: 2.039074
Train Epoch: 1 | Batch Status: 45000/50000 (90%) | Loss: 2.002053
Train Epoch: 1 | Batch Status: 45500/50000 (91%) | Loss: 1.671359
Train Epoch: 1 | Batch Status: 46000/50000 (92%) | Loss: 1.819428
Train Epoch: 1 | Batch Status: 46500/50000 (93%) | Loss: 1.925453
Train Epoch: 1 | Batch Status: 47000/50000 (94%) | Loss: 1.756307
Train Epoch: 1 | Batch Status: 47500/50000 (95%) | Loss: 1.858029
Train Epoch: 1 | Batch Status: 48000/50000 (96%) | Loss: 1.697147
Train Epoch: 1 | Batch Status: 48500/50000 (97%) | Loss: 1.683047
Train Epoch: 1 | Batch Status: 49000/50000 (98%) | Loss: 1.728243
Train Epoch: 1 | Batch Status: 49500/50000 (99%) | Loss: 1.851579
===========================
Test set: Average loss: 0.0363, Accuracy: 3365/10000 (34%)
Train Epoch: 2 | Batch Status: 0/50000 (0%) | Loss: 1.725205
Train Epoch: 2 | Batch Status: 500/50000 (1%) | Loss: 1.690922
Train Epoch: 2 | Batch Status: 1000/50000 (2%) | Loss: 1.641319
Train Epoch: 2 | Batch Status: 1500/50000 (3%) | Loss: 1.867115
Train Epoch: 2 | Batch Status: 2000/50000 (4%) | Loss: 1.914450
Train Epoch: 2 | Batch Status: 2500/50000 (5%) | Loss: 1.932380
Train Epoch: 2 | Batch Status: 3000/50000 (6%) | Loss: 1.810586
Train Epoch: 2 | Batch Status: 3500/50000 (7%) | Loss: 2.224813
Train Epoch: 2 | Batch Status: 4000/50000 (8%) | Loss: 1.782218
Train Epoch: 2 | Batch Status: 4500/50000 (9%) | Loss: 1.825632
Train Epoch: 2 | Batch Status: 5000/50000 (10%) | Loss: 1.669841
Train Epoch: 2 | Batch Status: 5500/50000 (11%) | Loss: 1.774749
Train Epoch: 2 | Batch Status: 6000/50000 (12%) | Loss: 2.060492
Train Epoch: 2 | Batch Status: 6500/50000 (13%) | Loss: 2.020693
Train Epoch: 2 | Batch Status: 7000/50000 (14%) | Loss: 1.663574
Train Epoch: 2 | Batch Status: 7500/50000 (15%) | Loss: 1.775136
Train Epoch: 2 | Batch Status: 8000/50000 (16%) | Loss: 1.776816
Train Epoch: 2 | Batch Status: 8500/50000 (17%) | Loss: 1.529867
Train Epoch: 2 | Batch Status: 9000/50000 (18%) | Loss: 1.689524
Train Epoch: 2 | Batch Status: 9500/50000 (19%) | Loss: 1.791537
Train Epoch: 2 | Batch Status: 10000/50000 (20%) | Loss: 1.816953
Train Epoch: 2 | Batch Status: 10500/50000 (21%) | Loss: 1.682496
Train Epoch: 2 | Batch Status: 11000/50000 (22%) | Loss: 1.741449
Train Epoch: 2 | Batch Status: 11500/50000 (23%) | Loss: 1.844448
Train Epoch: 2 | Batch Status: 12000/50000 (24%) | Loss: 1.858137
Train Epoch: 2 | Batch Status: 12500/50000 (25%) | Loss: 1.877445
Train Epoch: 2 | Batch Status: 13000/50000 (26%) | Loss: 1.753815
Train Epoch: 2 | Batch Status: 13500/50000 (27%) | Loss: 2.090965
Train Epoch: 2 | Batch Status: 14000/50000 (28%) | Loss: 1.898187
Train Epoch: 2 | Batch Status: 14500/50000 (29%) | Loss: 1.846180
Train Epoch: 2 | Batch Status: 15000/50000 (30%) | Loss: 1.594417
Train Epoch: 2 | Batch Status: 15500/50000 (31%) | Loss: 1.842082
Train Epoch: 2 | Batch Status: 16000/50000 (32%) | Loss: 1.804312
Train Epoch: 2 | Batch Status: 16500/50000 (33%) | Loss: 1.639538
Train Epoch: 2 | Batch Status: 17000/50000 (34%) | Loss: 1.879759
Train Epoch: 2 | Batch Status: 17500/50000 (35%) | Loss: 1.877845
Train Epoch: 2 | Batch Status: 18000/50000 (36%) | Loss: 1.648819
Train Epoch: 2 | Batch Status: 18500/50000 (37%) | Loss: 1.847155
Train Epoch: 2 | Batch Status: 19000/50000 (38%) | Loss: 1.823886
Train Epoch: 2 | Batch Status: 19500/50000 (39%) | Loss: 1.753661
Train Epoch: 2 | Batch Status: 20000/50000 (40%) | Loss: 1.699474
Train Epoch: 2 | Batch Status: 20500/50000 (41%) | Loss: 1.962556
Train Epoch: 2 | Batch Status: 21000/50000 (42%) | Loss: 1.622555
Train Epoch: 2 | Batch Status: 21500/50000 (43%) | Loss: 1.740715
Train Epoch: 2 | Batch Status: 22000/50000 (44%) | Loss: 1.647782
Train Epoch: 2 | Batch Status: 22500/50000 (45%) | Loss: 1.872705
Train Epoch: 2 | Batch Status: 23000/50000 (46%) | Loss: 2.019910
Train Epoch: 2 | Batch Status: 23500/50000 (47%) | Loss: 1.603308
Train Epoch: 2 | Batch Status: 24000/50000 (48%) | Loss: 2.025931
Train Epoch: 2 | Batch Status: 24500/50000 (49%) | Loss: 1.912077
Train Epoch: 2 | Batch Status: 25000/50000 (50%) | Loss: 1.928467
Train Epoch: 2 | Batch Status: 25500/50000 (51%) | Loss: 1.815359
Train Epoch: 2 | Batch Status: 26000/50000 (52%) | Loss: 1.981825
Train Epoch: 2 | Batch Status: 26500/50000 (53%) | Loss: 1.679863
Train Epoch: 2 | Batch Status: 27000/50000 (54%) | Loss: 1.571107
Train Epoch: 2 | Batch Status: 27500/50000 (55%) | Loss: 1.754611
Train Epoch: 2 | Batch Status: 28000/50000 (56%) | Loss: 1.744252
Train Epoch: 2 | Batch Status: 28500/50000 (57%) | Loss: 1.780448
Train Epoch: 2 | Batch Status: 29000/50000 (58%) | Loss: 2.016253
Train Epoch: 2 | Batch Status: 29500/50000 (59%) | Loss: 1.636006
Train Epoch: 2 | Batch Status: 30000/50000 (60%) | Loss: 1.705575
Train Epoch: 2 | Batch Status: 30500/50000 (61%) | Loss: 2.151510
Train Epoch: 2 | Batch Status: 31000/50000 (62%) | Loss: 1.919246
Train Epoch: 2 | Batch Status: 31500/50000 (63%) | Loss: 1.769660
Train Epoch: 2 | Batch Status: 32000/50000 (64%) | Loss: 1.474024
Train Epoch: 2 | Batch Status: 32500/50000 (65%) | Loss: 1.579773
Train Epoch: 2 | Batch Status: 33000/50000 (66%) | Loss: 1.665012
Train Epoch: 2 | Batch Status: 33500/50000 (67%) | Loss: 1.925456
Train Epoch: 2 | Batch Status: 34000/50000 (68%) | Loss: 1.682562
Train Epoch: 2 | Batch Status: 34500/50000 (69%) | Loss: 1.687774
Train Epoch: 2 | Batch Status: 35000/50000 (70%) | Loss: 1.903172
Train Epoch: 2 | Batch Status: 35500/50000 (71%) | Loss: 1.591886
Train Epoch: 2 | Batch Status: 36000/50000 (72%) | Loss: 1.505000
Train Epoch: 2 | Batch Status: 36500/50000 (73%) | Loss: 1.846029
Train Epoch: 2 | Batch Status: 37000/50000 (74%) | Loss: 1.621537
Train Epoch: 2 | Batch Status: 37500/50000 (75%) | Loss: 1.674948
Train Epoch: 2 | Batch Status: 38000/50000 (76%) | Loss: 1.421969
Train Epoch: 2 | Batch Status: 38500/50000 (77%) | Loss: 1.696380
Train Epoch: 2 | Batch Status: 39000/50000 (78%) | Loss: 1.572423
Train Epoch: 2 | Batch Status: 39500/50000 (79%) | Loss: 1.521675
Train Epoch: 2 | Batch Status: 40000/50000 (80%) | Loss: 1.659594
Train Epoch: 2 | Batch Status: 40500/50000 (81%) | Loss: 1.985753
Train Epoch: 2 | Batch Status: 41000/50000 (82%) | Loss: 1.793859
Train Epoch: 2 | Batch Status: 41500/50000 (83%) | Loss: 1.726932
Train Epoch: 2 | Batch Status: 42000/50000 (84%) | Loss: 1.665990
Train Epoch: 2 | Batch Status: 42500/50000 (85%) | Loss: 1.844720
Train Epoch: 2 | Batch Status: 43000/50000 (86%) | Loss: 1.681199
Train Epoch: 2 | Batch Status: 43500/50000 (87%) | Loss: 1.695129
Train Epoch: 2 | Batch Status: 44000/50000 (88%) | Loss: 1.810297
Train Epoch: 2 | Batch Status: 44500/50000 (89%) | Loss: 1.924766
Train Epoch: 2 | Batch Status: 45000/50000 (90%) | Loss: 1.648866
Train Epoch: 2 | Batch Status: 45500/50000 (91%) | Loss: 1.815055
Train Epoch: 2 | Batch Status: 46000/50000 (92%) | Loss: 1.816595
Train Epoch: 2 | Batch Status: 46500/50000 (93%) | Loss: 1.687417
Train Epoch: 2 | Batch Status: 47000/50000 (94%) | Loss: 1.667543
Train Epoch: 2 | Batch Status: 47500/50000 (95%) | Loss: 1.680392
Train Epoch: 2 | Batch Status: 48000/50000 (96%) | Loss: 1.455919
Train Epoch: 2 | Batch Status: 48500/50000 (97%) | Loss: 1.769703
Train Epoch: 2 | Batch Status: 49000/50000 (98%) | Loss: 1.963995
Train Epoch: 2 | Batch Status: 49500/50000 (99%) | Loss: 1.650082
===========================
Test set: Average loss: 0.0335, Accuracy: 3949/10000 (39%)
Train Epoch: 3 | Batch Status: 0/50000 (0%) | Loss: 1.554548
Train Epoch: 3 | Batch Status: 500/50000 (1%) | Loss: 1.607331
Train Epoch: 3 | Batch Status: 1000/50000 (2%) | Loss: 1.965468
Train Epoch: 3 | Batch Status: 1500/50000 (3%) | Loss: 1.472613
Train Epoch: 3 | Batch Status: 2000/50000 (4%) | Loss: 1.847007
Train Epoch: 3 | Batch Status: 2500/50000 (5%) | Loss: 1.722543
Train Epoch: 3 | Batch Status: 3000/50000 (6%) | Loss: 1.866580
Train Epoch: 3 | Batch Status: 3500/50000 (7%) | Loss: 1.752556
Train Epoch: 3 | Batch Status: 4000/50000 (8%) | Loss: 1.767413
Train Epoch: 3 | Batch Status: 4500/50000 (9%) | Loss: 1.788438
Train Epoch: 3 | Batch Status: 5000/50000 (10%) | Loss: 1.542054
Train Epoch: 3 | Batch Status: 5500/50000 (11%) | Loss: 1.702566
Train Epoch: 3 | Batch Status: 6000/50000 (12%) | Loss: 1.642337
Train Epoch: 3 | Batch Status: 6500/50000 (13%) | Loss: 1.659576
Train Epoch: 3 | Batch Status: 7000/50000 (14%) | Loss: 1.702328
Train Epoch: 3 | Batch Status: 7500/50000 (15%) | Loss: 1.567634
Train Epoch: 3 | Batch Status: 8000/50000 (16%) | Loss: 1.715549
Train Epoch: 3 | Batch Status: 8500/50000 (17%) | Loss: 1.562282
Train Epoch: 3 | Batch Status: 9000/50000 (18%) | Loss: 1.680778
Train Epoch: 3 | Batch Status: 9500/50000 (19%) | Loss: 1.750760
Train Epoch: 3 | Batch Status: 10000/50000 (20%) | Loss: 1.685149
Train Epoch: 3 | Batch Status: 10500/50000 (21%) | Loss: 1.715808
Train Epoch: 3 | Batch Status: 11000/50000 (22%) | Loss: 1.552358
Train Epoch: 3 | Batch Status: 11500/50000 (23%) | Loss: 1.390201
Train Epoch: 3 | Batch Status: 12000/50000 (24%) | Loss: 1.553566
Train Epoch: 3 | Batch Status: 12500/50000 (25%) | Loss: 1.593008
Train Epoch: 3 | Batch Status: 13000/50000 (26%) | Loss: 1.713692
Train Epoch: 3 | Batch Status: 13500/50000 (27%) | Loss: 1.499129
Train Epoch: 3 | Batch Status: 14000/50000 (28%) | Loss: 1.702059
Train Epoch: 3 | Batch Status: 14500/50000 (29%) | Loss: 1.657472
Train Epoch: 3 | Batch Status: 15000/50000 (30%) | Loss: 1.729217
Train Epoch: 3 | Batch Status: 15500/50000 (31%) | Loss: 1.631438
Train Epoch: 3 | Batch Status: 16000/50000 (32%) | Loss: 1.951836
Train Epoch: 3 | Batch Status: 16500/50000 (33%) | Loss: 2.027745
Train Epoch: 3 | Batch Status: 17000/50000 (34%) | Loss: 1.840118
Train Epoch: 3 | Batch Status: 17500/50000 (35%) | Loss: 1.615476
Train Epoch: 3 | Batch Status: 18000/50000 (36%) | Loss: 1.530296
Train Epoch: 3 | Batch Status: 18500/50000 (37%) | Loss: 1.825943
Train Epoch: 3 | Batch Status: 19000/50000 (38%) | Loss: 1.736305
Train Epoch: 3 | Batch Status: 19500/50000 (39%) | Loss: 1.589897
Train Epoch: 3 | Batch Status: 20000/50000 (40%) | Loss: 1.714262
Train Epoch: 3 | Batch Status: 20500/50000 (41%) | Loss: 1.861804
Train Epoch: 3 | Batch Status: 21000/50000 (42%) | Loss: 1.702979
Train Epoch: 3 | Batch Status: 21500/50000 (43%) | Loss: 1.658268
Train Epoch: 3 | Batch Status: 22000/50000 (44%) | Loss: 1.775440
Train Epoch: 3 | Batch Status: 22500/50000 (45%) | Loss: 1.682683
Train Epoch: 3 | Batch Status: 23000/50000 (46%) | Loss: 1.482317
Train Epoch: 3 | Batch Status: 23500/50000 (47%) | Loss: 1.577598
Train Epoch: 3 | Batch Status: 24000/50000 (48%) | Loss: 1.456940
Train Epoch: 3 | Batch Status: 24500/50000 (49%) | Loss: 1.715712
Train Epoch: 3 | Batch Status: 25000/50000 (50%) | Loss: 1.864707
Train Epoch: 3 | Batch Status: 25500/50000 (51%) | Loss: 1.869241
Train Epoch: 3 | Batch Status: 26000/50000 (52%) | Loss: 1.563475
Train Epoch: 3 | Batch Status: 26500/50000 (53%) | Loss: 1.787294
Train Epoch: 3 | Batch Status: 27000/50000 (54%) | Loss: 1.710690
Train Epoch: 3 | Batch Status: 27500/50000 (55%) | Loss: 1.398491
Train Epoch: 3 | Batch Status: 28000/50000 (56%) | Loss: 1.621918
Train Epoch: 3 | Batch Status: 28500/50000 (57%) | Loss: 1.856114
Train Epoch: 3 | Batch Status: 29000/50000 (58%) | Loss: 1.550666
Train Epoch: 3 | Batch Status: 29500/50000 (59%) | Loss: 1.803821
Train Epoch: 3 | Batch Status: 30000/50000 (60%) | Loss: 1.663117
Train Epoch: 3 | Batch Status: 30500/50000 (61%) | Loss: 1.900365
Train Epoch: 3 | Batch Status: 31000/50000 (62%) | Loss: 1.825365
Train Epoch: 3 | Batch Status: 31500/50000 (63%) | Loss: 1.896550
Train Epoch: 3 | Batch Status: 32000/50000 (64%) | Loss: 1.701376
Train Epoch: 3 | Batch Status: 32500/50000 (65%) | Loss: 1.769257
Train Epoch: 3 | Batch Status: 33000/50000 (66%) | Loss: 1.737017
Train Epoch: 3 | Batch Status: 33500/50000 (67%) | Loss: 1.755188
Train Epoch: 3 | Batch Status: 34000/50000 (68%) | Loss: 1.555652
Train Epoch: 3 | Batch Status: 34500/50000 (69%) | Loss: 1.819430
Train Epoch: 3 | Batch Status: 35000/50000 (70%) | Loss: 1.799355
Train Epoch: 3 | Batch Status: 35500/50000 (71%) | Loss: 1.465953
Train Epoch: 3 | Batch Status: 36000/50000 (72%) | Loss: 1.597577
Train Epoch: 3 | Batch Status: 36500/50000 (73%) | Loss: 1.744926
Train Epoch: 3 | Batch Status: 37000/50000 (74%) | Loss: 2.031130
Train Epoch: 3 | Batch Status: 37500/50000 (75%) | Loss: 1.578833
Train Epoch: 3 | Batch Status: 38000/50000 (76%) | Loss: 1.834866
Train Epoch: 3 | Batch Status: 38500/50000 (77%) | Loss: 1.584776
Train Epoch: 3 | Batch Status: 39000/50000 (78%) | Loss: 1.671542
Train Epoch: 3 | Batch Status: 39500/50000 (79%) | Loss: 1.701315
Train Epoch: 3 | Batch Status: 40000/50000 (80%) | Loss: 1.305912
Train Epoch: 3 | Batch Status: 40500/50000 (81%) | Loss: 1.452411
Train Epoch: 3 | Batch Status: 41000/50000 (82%) | Loss: 1.531587
Train Epoch: 3 | Batch Status: 41500/50000 (83%) | Loss: 1.501565
Train Epoch: 3 | Batch Status: 42000/50000 (84%) | Loss: 1.670327
Train Epoch: 3 | Batch Status: 42500/50000 (85%) | Loss: 1.697667
Train Epoch: 3 | Batch Status: 43000/50000 (86%) | Loss: 1.682536
Train Epoch: 3 | Batch Status: 43500/50000 (87%) | Loss: 1.558330
Train Epoch: 3 | Batch Status: 44000/50000 (88%) | Loss: 1.659899
Train Epoch: 3 | Batch Status: 44500/50000 (89%) | Loss: 1.487894
Train Epoch: 3 | Batch Status: 45000/50000 (90%) | Loss: 1.615816
Train Epoch: 3 | Batch Status: 45500/50000 (91%) | Loss: 1.384288
Train Epoch: 3 | Batch Status: 46000/50000 (92%) | Loss: 1.340137
Train Epoch: 3 | Batch Status: 46500/50000 (93%) | Loss: 1.521895
Train Epoch: 3 | Batch Status: 47000/50000 (94%) | Loss: 1.665667
Train Epoch: 3 | Batch Status: 47500/50000 (95%) | Loss: 1.320351
Train Epoch: 3 | Batch Status: 48000/50000 (96%) | Loss: 1.569734
Train Epoch: 3 | Batch Status: 48500/50000 (97%) | Loss: 1.629350
Train Epoch: 3 | Batch Status: 49000/50000 (98%) | Loss: 1.696789
Train Epoch: 3 | Batch Status: 49500/50000 (99%) | Loss: 1.564914
===========================
Test set: Average loss: 0.0323, Accuracy: 4207/10000 (42%)
Train Epoch: 4 | Batch Status: 0/50000 (0%) | Loss: 1.523216
Train Epoch: 4 | Batch Status: 500/50000 (1%) | Loss: 1.679904
Train Epoch: 4 | Batch Status: 1000/50000 (2%) | Loss: 1.435976
Train Epoch: 4 | Batch Status: 1500/50000 (3%) | Loss: 1.525869
Train Epoch: 4 | Batch Status: 2000/50000 (4%) | Loss: 1.462011
Train Epoch: 4 | Batch Status: 2500/50000 (5%) | Loss: 1.625124
Train Epoch: 4 | Batch Status: 3000/50000 (6%) | Loss: 1.719988
Train Epoch: 4 | Batch Status: 3500/50000 (7%) | Loss: 1.573516
Train Epoch: 4 | Batch Status: 4000/50000 (8%) | Loss: 1.404425
Train Epoch: 4 | Batch Status: 4500/50000 (9%) | Loss: 1.617122
Train Epoch: 4 | Batch Status: 5000/50000 (10%) | Loss: 1.291229
Train Epoch: 4 | Batch Status: 5500/50000 (11%) | Loss: 1.529075
Train Epoch: 4 | Batch Status: 6000/50000 (12%) | Loss: 1.461999
Train Epoch: 4 | Batch Status: 6500/50000 (13%) | Loss: 1.567406
Train Epoch: 4 | Batch Status: 7000/50000 (14%) | Loss: 1.603027
Train Epoch: 4 | Batch Status: 7500/50000 (15%) | Loss: 1.862623
Train Epoch: 4 | Batch Status: 8000/50000 (16%) | Loss: 1.700966
Train Epoch: 4 | Batch Status: 8500/50000 (17%) | Loss: 1.573964
Train Epoch: 4 | Batch Status: 9000/50000 (18%) | Loss: 1.597264
Train Epoch: 4 | Batch Status: 9500/50000 (19%) | Loss: 1.745172
Train Epoch: 4 | Batch Status: 10000/50000 (20%) | Loss: 1.436678
Train Epoch: 4 | Batch Status: 10500/50000 (21%) | Loss: 1.449118
Train Epoch: 4 | Batch Status: 11000/50000 (22%) | Loss: 1.363085
Train Epoch: 4 | Batch Status: 11500/50000 (23%) | Loss: 1.585526
Train Epoch: 4 | Batch Status: 12000/50000 (24%) | Loss: 1.750417
Train Epoch: 4 | Batch Status: 12500/50000 (25%) | Loss: 1.653231
Train Epoch: 4 | Batch Status: 13000/50000 (26%) | Loss: 1.779195
Train Epoch: 4 | Batch Status: 13500/50000 (27%) | Loss: 1.773649
Train Epoch: 4 | Batch Status: 14000/50000 (28%) | Loss: 1.583885
Train Epoch: 4 | Batch Status: 14500/50000 (29%) | Loss: 1.536166
Train Epoch: 4 | Batch Status: 15000/50000 (30%) | Loss: 1.482655
Train Epoch: 4 | Batch Status: 15500/50000 (31%) | Loss: 1.488769
Train Epoch: 4 | Batch Status: 16000/50000 (32%) | Loss: 1.438621
Train Epoch: 4 | Batch Status: 16500/50000 (33%) | Loss: 1.432055
Train Epoch: 4 | Batch Status: 17000/50000 (34%) | Loss: 1.733367
Train Epoch: 4 | Batch Status: 17500/50000 (35%) | Loss: 1.511536
Train Epoch: 4 | Batch Status: 18000/50000 (36%) | Loss: 1.506607
Train Epoch: 4 | Batch Status: 18500/50000 (37%) | Loss: 1.446141
Train Epoch: 4 | Batch Status: 19000/50000 (38%) | Loss: 1.654448
Train Epoch: 4 | Batch Status: 19500/50000 (39%) | Loss: 1.563035
Train Epoch: 4 | Batch Status: 20000/50000 (40%) | Loss: 1.408949
Train Epoch: 4 | Batch Status: 20500/50000 (41%) | Loss: 1.449776
Train Epoch: 4 | Batch Status: 21000/50000 (42%) | Loss: 1.654020
Train Epoch: 4 | Batch Status: 21500/50000 (43%) | Loss: 1.396821
Train Epoch: 4 | Batch Status: 22000/50000 (44%) | Loss: 1.612035
Train Epoch: 4 | Batch Status: 22500/50000 (45%) | Loss: 1.666113
Train Epoch: 4 | Batch Status: 23000/50000 (46%) | Loss: 1.489748
Train Epoch: 4 | Batch Status: 23500/50000 (47%) | Loss: 1.647463
Train Epoch: 4 | Batch Status: 24000/50000 (48%) | Loss: 1.673048
Train Epoch: 4 | Batch Status: 24500/50000 (49%) | Loss: 1.681192
Train Epoch: 4 | Batch Status: 25000/50000 (50%) | Loss: 1.494779
Train Epoch: 4 | Batch Status: 25500/50000 (51%) | Loss: 1.568911
Train Epoch: 4 | Batch Status: 26000/50000 (52%) | Loss: 1.471634
Train Epoch: 4 | Batch Status: 26500/50000 (53%) | Loss: 1.529602
Train Epoch: 4 | Batch Status: 27000/50000 (54%) | Loss: 1.886674
Train Epoch: 4 | Batch Status: 27500/50000 (55%) | Loss: 1.750676
Train Epoch: 4 | Batch Status: 28000/50000 (56%) | Loss: 1.468764
Train Epoch: 4 | Batch Status: 28500/50000 (57%) | Loss: 1.540622
Train Epoch: 4 | Batch Status: 29000/50000 (58%) | Loss: 1.789050
Train Epoch: 4 | Batch Status: 29500/50000 (59%) | Loss: 1.333626
Train Epoch: 4 | Batch Status: 30000/50000 (60%) | Loss: 1.412051
Train Epoch: 4 | Batch Status: 30500/50000 (61%) | Loss: 1.631136
Train Epoch: 4 | Batch Status: 31000/50000 (62%) | Loss: 1.668067
Train Epoch: 4 | Batch Status: 31500/50000 (63%) | Loss: 1.273194
Train Epoch: 4 | Batch Status: 32000/50000 (64%) | Loss: 1.495729
Train Epoch: 4 | Batch Status: 32500/50000 (65%) | Loss: 1.516832
Train Epoch: 4 | Batch Status: 33000/50000 (66%) | Loss: 1.481310
Train Epoch: 4 | Batch Status: 33500/50000 (67%) | Loss: 2.005491
Train Epoch: 4 | Batch Status: 34000/50000 (68%) | Loss: 1.657755
Train Epoch: 4 | Batch Status: 34500/50000 (69%) | Loss: 1.457682
Train Epoch: 4 | Batch Status: 35000/50000 (70%) | Loss: 1.455632
Train Epoch: 4 | Batch Status: 35500/50000 (71%) | Loss: 1.697754
Train Epoch: 4 | Batch Status: 36000/50000 (72%) | Loss: 1.404963
Train Epoch: 4 | Batch Status: 36500/50000 (73%) | Loss: 1.597792
Train Epoch: 4 | Batch Status: 37000/50000 (74%) | Loss: 1.621331
Train Epoch: 4 | Batch Status: 37500/50000 (75%) | Loss: 1.547942
Train Epoch: 4 | Batch Status: 38000/50000 (76%) | Loss: 1.682565
Train Epoch: 4 | Batch Status: 38500/50000 (77%) | Loss: 1.622169
Train Epoch: 4 | Batch Status: 39000/50000 (78%) | Loss: 1.640330
Train Epoch: 4 | Batch Status: 39500/50000 (79%) | Loss: 1.431819
Train Epoch: 4 | Batch Status: 40000/50000 (80%) | Loss: 1.556864
Train Epoch: 4 | Batch Status: 40500/50000 (81%) | Loss: 1.658859
Train Epoch: 4 | Batch Status: 41000/50000 (82%) | Loss: 1.715026
Train Epoch: 4 | Batch Status: 41500/50000 (83%) | Loss: 1.737730
Train Epoch: 4 | Batch Status: 42000/50000 (84%) | Loss: 1.237931
Train Epoch: 4 | Batch Status: 42500/50000 (85%) | Loss: 1.349263
Train Epoch: 4 | Batch Status: 43000/50000 (86%) | Loss: 1.571223
Train Epoch: 4 | Batch Status: 43500/50000 (87%) | Loss: 1.441681
Train Epoch: 4 | Batch Status: 44000/50000 (88%) | Loss: 1.334654
Train Epoch: 4 | Batch Status: 44500/50000 (89%) | Loss: 1.678486
Train Epoch: 4 | Batch Status: 45000/50000 (90%) | Loss: 1.397450
Train Epoch: 4 | Batch Status: 45500/50000 (91%) | Loss: 1.632686
Train Epoch: 4 | Batch Status: 46000/50000 (92%) | Loss: 1.597266
Train Epoch: 4 | Batch Status: 46500/50000 (93%) | Loss: 1.593452
Train Epoch: 4 | Batch Status: 47000/50000 (94%) | Loss: 1.567113
Train Epoch: 4 | Batch Status: 47500/50000 (95%) | Loss: 1.848821
Train Epoch: 4 | Batch Status: 48000/50000 (96%) | Loss: 1.449719
Train Epoch: 4 | Batch Status: 48500/50000 (97%) | Loss: 1.482795
Train Epoch: 4 | Batch Status: 49000/50000 (98%) | Loss: 1.444067
Train Epoch: 4 | Batch Status: 49500/50000 (99%) | Loss: 1.546082
===========================
Test set: Average loss: 0.0317, Accuracy: 4320/10000 (43%)
Train Epoch: 5 | Batch Status: 0/50000 (0%) | Loss: 2.070221
Train Epoch: 5 | Batch Status: 500/50000 (1%) | Loss: 1.412957
Train Epoch: 5 | Batch Status: 1000/50000 (2%) | Loss: 1.481404
Train Epoch: 5 | Batch Status: 1500/50000 (3%) | Loss: 1.514722
Train Epoch: 5 | Batch Status: 2000/50000 (4%) | Loss: 1.461223
Train Epoch: 5 | Batch Status: 2500/50000 (5%) | Loss: 1.684770
Train Epoch: 5 | Batch Status: 3000/50000 (6%) | Loss: 1.465889
Train Epoch: 5 | Batch Status: 3500/50000 (7%) | Loss: 1.775378
Train Epoch: 5 | Batch Status: 4000/50000 (8%) | Loss: 1.149485
Train Epoch: 5 | Batch Status: 4500/50000 (9%) | Loss: 1.618577
Train Epoch: 5 | Batch Status: 5000/50000 (10%) | Loss: 1.792500
Train Epoch: 5 | Batch Status: 5500/50000 (11%) | Loss: 1.168486
Train Epoch: 5 | Batch Status: 6000/50000 (12%) | Loss: 1.599952
Train Epoch: 5 | Batch Status: 6500/50000 (13%) | Loss: 1.347074
Train Epoch: 5 | Batch Status: 7000/50000 (14%) | Loss: 1.817937
Train Epoch: 5 | Batch Status: 7500/50000 (15%) | Loss: 1.591971
Train Epoch: 5 | Batch Status: 8000/50000 (16%) | Loss: 1.289740
Train Epoch: 5 | Batch Status: 8500/50000 (17%) | Loss: 1.521550
Train Epoch: 5 | Batch Status: 9000/50000 (18%) | Loss: 1.524179
Train Epoch: 5 | Batch Status: 9500/50000 (19%) | Loss: 1.665891
Train Epoch: 5 | Batch Status: 10000/50000 (20%) | Loss: 1.504113
Train Epoch: 5 | Batch Status: 10500/50000 (21%) | Loss: 1.541767
Train Epoch: 5 | Batch Status: 11000/50000 (22%) | Loss: 1.853924
Train Epoch: 5 | Batch Status: 11500/50000 (23%) | Loss: 1.654423
Train Epoch: 5 | Batch Status: 12000/50000 (24%) | Loss: 1.391741
Train Epoch: 5 | Batch Status: 12500/50000 (25%) | Loss: 1.439986
Train Epoch: 5 | Batch Status: 13000/50000 (26%) | Loss: 1.588511
Train Epoch: 5 | Batch Status: 13500/50000 (27%) | Loss: 1.745847
Train Epoch: 5 | Batch Status: 14000/50000 (28%) | Loss: 1.562533
Train Epoch: 5 | Batch Status: 14500/50000 (29%) | Loss: 1.447875
Train Epoch: 5 | Batch Status: 15000/50000 (30%) | Loss: 1.351276
Train Epoch: 5 | Batch Status: 15500/50000 (31%) | Loss: 1.656896
Train Epoch: 5 | Batch Status: 16000/50000 (32%) | Loss: 1.324456
Train Epoch: 5 | Batch Status: 16500/50000 (33%) | Loss: 1.354311
Train Epoch: 5 | Batch Status: 17000/50000 (34%) | Loss: 1.779609
Train Epoch: 5 | Batch Status: 17500/50000 (35%) | Loss: 1.300896
Train Epoch: 5 | Batch Status: 18000/50000 (36%) | Loss: 1.395349
Train Epoch: 5 | Batch Status: 18500/50000 (37%) | Loss: 1.640097
Train Epoch: 5 | Batch Status: 19000/50000 (38%) | Loss: 1.656469
Train Epoch: 5 | Batch Status: 19500/50000 (39%) | Loss: 1.621350
Train Epoch: 5 | Batch Status: 20000/50000 (40%) | Loss: 1.469339
Train Epoch: 5 | Batch Status: 20500/50000 (41%) | Loss: 1.595617
Train Epoch: 5 | Batch Status: 21000/50000 (42%) | Loss: 1.649909
Train Epoch: 5 | Batch Status: 21500/50000 (43%) | Loss: 1.665491
Train Epoch: 5 | Batch Status: 22000/50000 (44%) | Loss: 1.642666
Train Epoch: 5 | Batch Status: 22500/50000 (45%) | Loss: 1.316798
Train Epoch: 5 | Batch Status: 23000/50000 (46%) | Loss: 1.426536
Train Epoch: 5 | Batch Status: 23500/50000 (47%) | Loss: 1.469971
Train Epoch: 5 | Batch Status: 24000/50000 (48%) | Loss: 1.707075
Train Epoch: 5 | Batch Status: 24500/50000 (49%) | Loss: 1.555358
Train Epoch: 5 | Batch Status: 25000/50000 (50%) | Loss: 1.573900
Train Epoch: 5 | Batch Status: 25500/50000 (51%) | Loss: 1.486640
Train Epoch: 5 | Batch Status: 26000/50000 (52%) | Loss: 1.585356
Train Epoch: 5 | Batch Status: 26500/50000 (53%) | Loss: 1.810102
Train Epoch: 5 | Batch Status: 27000/50000 (54%) | Loss: 1.780408
Train Epoch: 5 | Batch Status: 27500/50000 (55%) | Loss: 1.473457
Train Epoch: 5 | Batch Status: 28000/50000 (56%) | Loss: 1.836347
Train Epoch: 5 | Batch Status: 28500/50000 (57%) | Loss: 1.739414
Train Epoch: 5 | Batch Status: 29000/50000 (58%) | Loss: 1.625105
Train Epoch: 5 | Batch Status: 29500/50000 (59%) | Loss: 1.440914
Train Epoch: 5 | Batch Status: 30000/50000 (60%) | Loss: 1.432192
Train Epoch: 5 | Batch Status: 30500/50000 (61%) | Loss: 1.696207
Train Epoch: 5 | Batch Status: 31000/50000 (62%) | Loss: 1.438973
Train Epoch: 5 | Batch Status: 31500/50000 (63%) | Loss: 1.453859
Train Epoch: 5 | Batch Status: 32000/50000 (64%) | Loss: 1.329051
Train Epoch: 5 | Batch Status: 32500/50000 (65%) | Loss: 1.211526
Train Epoch: 5 | Batch Status: 33000/50000 (66%) | Loss: 1.369125
Train Epoch: 5 | Batch Status: 33500/50000 (67%) | Loss: 1.818081
Train Epoch: 5 | Batch Status: 34000/50000 (68%) | Loss: 1.581870
Train Epoch: 5 | Batch Status: 34500/50000 (69%) | Loss: 1.707785
Train Epoch: 5 | Batch Status: 35000/50000 (70%) | Loss: 1.491172
Train Epoch: 5 | Batch Status: 35500/50000 (71%) | Loss: 1.497230
Train Epoch: 5 | Batch Status: 36000/50000 (72%) | Loss: 1.583376
Train Epoch: 5 | Batch Status: 36500/50000 (73%) | Loss: 1.384699
Train Epoch: 5 | Batch Status: 37000/50000 (74%) | Loss: 1.717561
Train Epoch: 5 | Batch Status: 37500/50000 (75%) | Loss: 1.809301
Train Epoch: 5 | Batch Status: 38000/50000 (76%) | Loss: 1.529655
Train Epoch: 5 | Batch Status: 38500/50000 (77%) | Loss: 1.618351
Train Epoch: 5 | Batch Status: 39000/50000 (78%) | Loss: 1.492666
Train Epoch: 5 | Batch Status: 39500/50000 (79%) | Loss: 1.790830
Train Epoch: 5 | Batch Status: 40000/50000 (80%) | Loss: 1.392904
Train Epoch: 5 | Batch Status: 40500/50000 (81%) | Loss: 1.846031
Train Epoch: 5 | Batch Status: 41000/50000 (82%) | Loss: 1.444642
Train Epoch: 5 | Batch Status: 41500/50000 (83%) | Loss: 1.346643
Train Epoch: 5 | Batch Status: 42000/50000 (84%) | Loss: 1.627847
Train Epoch: 5 | Batch Status: 42500/50000 (85%) | Loss: 1.730883
Train Epoch: 5 | Batch Status: 43000/50000 (86%) | Loss: 1.442618
Train Epoch: 5 | Batch Status: 43500/50000 (87%) | Loss: 1.335858
Train Epoch: 5 | Batch Status: 44000/50000 (88%) | Loss: 1.383894
Train Epoch: 5 | Batch Status: 44500/50000 (89%) | Loss: 1.520119
Train Epoch: 5 | Batch Status: 45000/50000 (90%) | Loss: 1.283693
Train Epoch: 5 | Batch Status: 45500/50000 (91%) | Loss: 1.505282
Train Epoch: 5 | Batch Status: 46000/50000 (92%) | Loss: 1.037320
Train Epoch: 5 | Batch Status: 46500/50000 (93%) | Loss: 1.207652
Train Epoch: 5 | Batch Status: 47000/50000 (94%) | Loss: 1.261377
Train Epoch: 5 | Batch Status: 47500/50000 (95%) | Loss: 1.645064
Train Epoch: 5 | Batch Status: 48000/50000 (96%) | Loss: 1.630996
Train Epoch: 5 | Batch Status: 48500/50000 (97%) | Loss: 1.399808
Train Epoch: 5 | Batch Status: 49000/50000 (98%) | Loss: 1.469568
Train Epoch: 5 | Batch Status: 49500/50000 (99%) | Loss: 1.597197
===========================
Test set: Average loss: 0.0305, Accuracy: 4495/10000 (45%)
Train Epoch: 6 | Batch Status: 0/50000 (0%) | Loss: 1.748882
Train Epoch: 6 | Batch Status: 500/50000 (1%) | Loss: 1.562962
Train Epoch: 6 | Batch Status: 1000/50000 (2%) | Loss: 1.304953
Train Epoch: 6 | Batch Status: 1500/50000 (3%) | Loss: 1.377832
Train Epoch: 6 | Batch Status: 2000/50000 (4%) | Loss: 1.425503
Train Epoch: 6 | Batch Status: 2500/50000 (5%) | Loss: 1.423137
Train Epoch: 6 | Batch Status: 3000/50000 (6%) | Loss: 1.309053
Train Epoch: 6 | Batch Status: 3500/50000 (7%) | Loss: 1.547056
Train Epoch: 6 | Batch Status: 4000/50000 (8%) | Loss: 1.412654
Train Epoch: 6 | Batch Status: 4500/50000 (9%) | Loss: 1.448599
Train Epoch: 6 | Batch Status: 5000/50000 (10%) | Loss: 1.398573
Train Epoch: 6 | Batch Status: 5500/50000 (11%) | Loss: 1.555349
Train Epoch: 6 | Batch Status: 6000/50000 (12%) | Loss: 1.400668
Train Epoch: 6 | Batch Status: 6500/50000 (13%) | Loss: 1.410115
Train Epoch: 6 | Batch Status: 7000/50000 (14%) | Loss: 1.505261
Train Epoch: 6 | Batch Status: 7500/50000 (15%) | Loss: 1.323009
Train Epoch: 6 | Batch Status: 8000/50000 (16%) | Loss: 1.746725
Train Epoch: 6 | Batch Status: 8500/50000 (17%) | Loss: 1.513857
Train Epoch: 6 | Batch Status: 9000/50000 (18%) | Loss: 1.252447
Train Epoch: 6 | Batch Status: 9500/50000 (19%) | Loss: 1.262275
Train Epoch: 6 | Batch Status: 10000/50000 (20%) | Loss: 1.791681
Train Epoch: 6 | Batch Status: 10500/50000 (21%) | Loss: 1.579397
Train Epoch: 6 | Batch Status: 11000/50000 (22%) | Loss: 1.538674
Train Epoch: 6 | Batch Status: 11500/50000 (23%) | Loss: 1.344683
Train Epoch: 6 | Batch Status: 12000/50000 (24%) | Loss: 1.469114
Train Epoch: 6 | Batch Status: 12500/50000 (25%) | Loss: 1.441337
Train Epoch: 6 | Batch Status: 13000/50000 (26%) | Loss: 1.323505
Train Epoch: 6 | Batch Status: 13500/50000 (27%) | Loss: 1.475913
Train Epoch: 6 | Batch Status: 14000/50000 (28%) | Loss: 1.318627
Train Epoch: 6 | Batch Status: 14500/50000 (29%) | Loss: 1.465369
Train Epoch: 6 | Batch Status: 15000/50000 (30%) | Loss: 1.304634
Train Epoch: 6 | Batch Status: 15500/50000 (31%) | Loss: 1.266232
Train Epoch: 6 | Batch Status: 16000/50000 (32%) | Loss: 1.513910
Train Epoch: 6 | Batch Status: 16500/50000 (33%) | Loss: 1.261921
Train Epoch: 6 | Batch Status: 17000/50000 (34%) | Loss: 1.320604
Train Epoch: 6 | Batch Status: 17500/50000 (35%) | Loss: 1.744250
Train Epoch: 6 | Batch Status: 18000/50000 (36%) | Loss: 1.351349
Train Epoch: 6 | Batch Status: 18500/50000 (37%) | Loss: 1.297133
Train Epoch: 6 | Batch Status: 19000/50000 (38%) | Loss: 1.553308
Train Epoch: 6 | Batch Status: 19500/50000 (39%) | Loss: 1.502329
Train Epoch: 6 | Batch Status: 20000/50000 (40%) | Loss: 1.406582
Train Epoch: 6 | Batch Status: 20500/50000 (41%) | Loss: 1.598224
Train Epoch: 6 | Batch Status: 21000/50000 (42%) | Loss: 1.512024
Train Epoch: 6 | Batch Status: 21500/50000 (43%) | Loss: 1.715929
Train Epoch: 6 | Batch Status: 22000/50000 (44%) | Loss: 1.594268
Train Epoch: 6 | Batch Status: 22500/50000 (45%) | Loss: 1.339259
Train Epoch: 6 | Batch Status: 23000/50000 (46%) | Loss: 1.380494
Train Epoch: 6 | Batch Status: 23500/50000 (47%) | Loss: 1.362838
Train Epoch: 6 | Batch Status: 24000/50000 (48%) | Loss: 1.498852
Train Epoch: 6 | Batch Status: 24500/50000 (49%) | Loss: 1.487684
Train Epoch: 6 | Batch Status: 25000/50000 (50%) | Loss: 1.655687
Train Epoch: 6 | Batch Status: 25500/50000 (51%) | Loss: 1.351408
Train Epoch: 6 | Batch Status: 26000/50000 (52%) | Loss: 1.472888
Train Epoch: 6 | Batch Status: 26500/50000 (53%) | Loss: 1.381280
Train Epoch: 6 | Batch Status: 27000/50000 (54%) | Loss: 1.612155
Train Epoch: 6 | Batch Status: 27500/50000 (55%) | Loss: 1.325075
Train Epoch: 6 | Batch Status: 28000/50000 (56%) | Loss: 1.244259
Train Epoch: 6 | Batch Status: 28500/50000 (57%) | Loss: 1.388186
Train Epoch: 6 | Batch Status: 29000/50000 (58%) | Loss: 1.597838
Train Epoch: 6 | Batch Status: 29500/50000 (59%) | Loss: 1.722281
Train Epoch: 6 | Batch Status: 30000/50000 (60%) | Loss: 1.500449
Train Epoch: 6 | Batch Status: 30500/50000 (61%) | Loss: 1.572149
Train Epoch: 6 | Batch Status: 31000/50000 (62%) | Loss: 1.385941
Train Epoch: 6 | Batch Status: 31500/50000 (63%) | Loss: 1.378388
Train Epoch: 6 | Batch Status: 32000/50000 (64%) | Loss: 1.363358
Train Epoch: 6 | Batch Status: 32500/50000 (65%) | Loss: 1.317546
Train Epoch: 6 | Batch Status: 33000/50000 (66%) | Loss: 1.482623
Train Epoch: 6 | Batch Status: 33500/50000 (67%) | Loss: 1.282279
Train Epoch: 6 | Batch Status: 34000/50000 (68%) | Loss: 1.776104
Train Epoch: 6 | Batch Status: 34500/50000 (69%) | Loss: 1.358479
Train Epoch: 6 | Batch Status: 35000/50000 (70%) | Loss: 1.363629
Train Epoch: 6 | Batch Status: 35500/50000 (71%) | Loss: 1.502507
Train Epoch: 6 | Batch Status: 36000/50000 (72%) | Loss: 1.634139
Train Epoch: 6 | Batch Status: 36500/50000 (73%) | Loss: 1.620142
Train Epoch: 6 | Batch Status: 37000/50000 (74%) | Loss: 1.341292
Train Epoch: 6 | Batch Status: 37500/50000 (75%) | Loss: 1.280996
Train Epoch: 6 | Batch Status: 38000/50000 (76%) | Loss: 1.566507
Train Epoch: 6 | Batch Status: 38500/50000 (77%) | Loss: 1.334921
Train Epoch: 6 | Batch Status: 39000/50000 (78%) | Loss: 1.588790
Train Epoch: 6 | Batch Status: 39500/50000 (79%) | Loss: 1.446336
Train Epoch: 6 | Batch Status: 40000/50000 (80%) | Loss: 1.358768
Train Epoch: 6 | Batch Status: 40500/50000 (81%) | Loss: 1.308481
Train Epoch: 6 | Batch Status: 41000/50000 (82%) | Loss: 1.430712
Train Epoch: 6 | Batch Status: 41500/50000 (83%) | Loss: 1.538562
Train Epoch: 6 | Batch Status: 42000/50000 (84%) | Loss: 1.579042
Train Epoch: 6 | Batch Status: 42500/50000 (85%) | Loss: 1.316051
Train Epoch: 6 | Batch Status: 43000/50000 (86%) | Loss: 1.360927
Train Epoch: 6 | Batch Status: 43500/50000 (87%) | Loss: 1.286301
Train Epoch: 6 | Batch Status: 44000/50000 (88%) | Loss: 1.407606
Train Epoch: 6 | Batch Status: 44500/50000 (89%) | Loss: 1.322438
Train Epoch: 6 | Batch Status: 45000/50000 (90%) | Loss: 1.544184
Train Epoch: 6 | Batch Status: 45500/50000 (91%) | Loss: 1.395002
Train Epoch: 6 | Batch Status: 46000/50000 (92%) | Loss: 1.505281
Train Epoch: 6 | Batch Status: 46500/50000 (93%) | Loss: 1.248829
Train Epoch: 6 | Batch Status: 47000/50000 (94%) | Loss: 1.726754
Train Epoch: 6 | Batch Status: 47500/50000 (95%) | Loss: 1.681255
Train Epoch: 6 | Batch Status: 48000/50000 (96%) | Loss: 1.550664
Train Epoch: 6 | Batch Status: 48500/50000 (97%) | Loss: 1.348505
Train Epoch: 6 | Batch Status: 49000/50000 (98%) | Loss: 1.411774
Train Epoch: 6 | Batch Status: 49500/50000 (99%) | Loss: 1.484563
===========================
Test set: Average loss: 0.0307, Accuracy: 4637/10000 (46%)
Train Epoch: 7 | Batch Status: 0/50000 (0%) | Loss: 1.407814
Train Epoch: 7 | Batch Status: 500/50000 (1%) | Loss: 1.363074
Train Epoch: 7 | Batch Status: 1000/50000 (2%) | Loss: 1.323917
Train Epoch: 7 | Batch Status: 1500/50000 (3%) | Loss: 1.207579
Train Epoch: 7 | Batch Status: 2000/50000 (4%) | Loss: 1.342503
Train Epoch: 7 | Batch Status: 2500/50000 (5%) | Loss: 1.259142
Train Epoch: 7 | Batch Status: 3000/50000 (6%) | Loss: 1.324044
Train Epoch: 7 | Batch Status: 3500/50000 (7%) | Loss: 1.388356
Train Epoch: 7 | Batch Status: 4000/50000 (8%) | Loss: 1.536059
Train Epoch: 7 | Batch Status: 4500/50000 (9%) | Loss: 1.479334
Train Epoch: 7 | Batch Status: 5000/50000 (10%) | Loss: 1.363794
Train Epoch: 7 | Batch Status: 5500/50000 (11%) | Loss: 1.519238
Train Epoch: 7 | Batch Status: 6000/50000 (12%) | Loss: 1.380227
Train Epoch: 7 | Batch Status: 6500/50000 (13%) | Loss: 1.339850
Train Epoch: 7 | Batch Status: 7000/50000 (14%) | Loss: 1.689961
Train Epoch: 7 | Batch Status: 7500/50000 (15%) | Loss: 1.392854
Train Epoch: 7 | Batch Status: 8000/50000 (16%) | Loss: 1.344524
Train Epoch: 7 | Batch Status: 8500/50000 (17%) | Loss: 1.411460
Train Epoch: 7 | Batch Status: 9000/50000 (18%) | Loss: 1.788046
Train Epoch: 7 | Batch Status: 9500/50000 (19%) | Loss: 1.402065
Train Epoch: 7 | Batch Status: 10000/50000 (20%) | Loss: 1.320393
Train Epoch: 7 | Batch Status: 10500/50000 (21%) | Loss: 1.219924
Train Epoch: 7 | Batch Status: 11000/50000 (22%) | Loss: 1.323237
Train Epoch: 7 | Batch Status: 11500/50000 (23%) | Loss: 1.721788
Train Epoch: 7 | Batch Status: 12000/50000 (24%) | Loss: 1.508190
Train Epoch: 7 | Batch Status: 12500/50000 (25%) | Loss: 1.391588
Train Epoch: 7 | Batch Status: 13000/50000 (26%) | Loss: 1.296960
Train Epoch: 7 | Batch Status: 13500/50000 (27%) | Loss: 1.373983
Train Epoch: 7 | Batch Status: 14000/50000 (28%) | Loss: 1.655529
Train Epoch: 7 | Batch Status: 14500/50000 (29%) | Loss: 1.048752
Train Epoch: 7 | Batch Status: 15000/50000 (30%) | Loss: 1.588992
Train Epoch: 7 | Batch Status: 15500/50000 (31%) | Loss: 1.552361
Train Epoch: 7 | Batch Status: 16000/50000 (32%) | Loss: 1.282272
Train Epoch: 7 | Batch Status: 16500/50000 (33%) | Loss: 1.130308
Train Epoch: 7 | Batch Status: 17000/50000 (34%) | Loss: 1.542151
Train Epoch: 7 | Batch Status: 17500/50000 (35%) | Loss: 1.737612
Train Epoch: 7 | Batch Status: 18000/50000 (36%) | Loss: 1.348980
Train Epoch: 7 | Batch Status: 18500/50000 (37%) | Loss: 1.131197
Train Epoch: 7 | Batch Status: 19000/50000 (38%) | Loss: 1.459149
Train Epoch: 7 | Batch Status: 19500/50000 (39%) | Loss: 1.322903
Train Epoch: 7 | Batch Status: 20000/50000 (40%) | Loss: 1.181113
Train Epoch: 7 | Batch Status: 20500/50000 (41%) | Loss: 1.448473
Train Epoch: 7 | Batch Status: 21000/50000 (42%) | Loss: 1.523440
Train Epoch: 7 | Batch Status: 21500/50000 (43%) | Loss: 1.280477
Train Epoch: 7 | Batch Status: 22000/50000 (44%) | Loss: 1.197843
Train Epoch: 7 | Batch Status: 22500/50000 (45%) | Loss: 1.537267
Train Epoch: 7 | Batch Status: 23000/50000 (46%) | Loss: 1.269421
Train Epoch: 7 | Batch Status: 23500/50000 (47%) | Loss: 1.397631
Train Epoch: 7 | Batch Status: 24000/50000 (48%) | Loss: 1.602262
Train Epoch: 7 | Batch Status: 24500/50000 (49%) | Loss: 1.396561
Train Epoch: 7 | Batch Status: 25000/50000 (50%) | Loss: 1.705571
Train Epoch: 7 | Batch Status: 25500/50000 (51%) | Loss: 1.654305
Train Epoch: 7 | Batch Status: 26000/50000 (52%) | Loss: 1.226056
Train Epoch: 7 | Batch Status: 26500/50000 (53%) | Loss: 1.253545
Train Epoch: 7 | Batch Status: 27000/50000 (54%) | Loss: 1.619253
Train Epoch: 7 | Batch Status: 27500/50000 (55%) | Loss: 1.497837
Train Epoch: 7 | Batch Status: 28000/50000 (56%) | Loss: 1.401160
Train Epoch: 7 | Batch Status: 28500/50000 (57%) | Loss: 1.588566
Train Epoch: 7 | Batch Status: 29000/50000 (58%) | Loss: 1.497738
Train Epoch: 7 | Batch Status: 29500/50000 (59%) | Loss: 1.618085
Train Epoch: 7 | Batch Status: 30000/50000 (60%) | Loss: 1.431337
Train Epoch: 7 | Batch Status: 30500/50000 (61%) | Loss: 1.592575
Train Epoch: 7 | Batch Status: 31000/50000 (62%) | Loss: 1.475086
Train Epoch: 7 | Batch Status: 31500/50000 (63%) | Loss: 1.312718
Train Epoch: 7 | Batch Status: 32000/50000 (64%) | Loss: 1.393072
Train Epoch: 7 | Batch Status: 32500/50000 (65%) | Loss: 1.419333
Train Epoch: 7 | Batch Status: 33000/50000 (66%) | Loss: 1.183939
Train Epoch: 7 | Batch Status: 33500/50000 (67%) | Loss: 1.557632
Train Epoch: 7 | Batch Status: 34000/50000 (68%) | Loss: 1.547253
Train Epoch: 7 | Batch Status: 34500/50000 (69%) | Loss: 1.325757
Train Epoch: 7 | Batch Status: 35000/50000 (70%) | Loss: 1.491250
Train Epoch: 7 | Batch Status: 35500/50000 (71%) | Loss: 1.489703
Train Epoch: 7 | Batch Status: 36000/50000 (72%) | Loss: 1.665197
Train Epoch: 7 | Batch Status: 36500/50000 (73%) | Loss: 1.216421
Train Epoch: 7 | Batch Status: 37000/50000 (74%) | Loss: 1.652310
Train Epoch: 7 | Batch Status: 37500/50000 (75%) | Loss: 1.427461
Train Epoch: 7 | Batch Status: 38000/50000 (76%) | Loss: 1.272539
Train Epoch: 7 | Batch Status: 38500/50000 (77%) | Loss: 1.340435
Train Epoch: 7 | Batch Status: 39000/50000 (78%) | Loss: 1.563349
Train Epoch: 7 | Batch Status: 39500/50000 (79%) | Loss: 1.358799
Train Epoch: 7 | Batch Status: 40000/50000 (80%) | Loss: 1.396223
Train Epoch: 7 | Batch Status: 40500/50000 (81%) | Loss: 1.278827
Train Epoch: 7 | Batch Status: 41000/50000 (82%) | Loss: 1.127343
Train Epoch: 7 | Batch Status: 41500/50000 (83%) | Loss: 1.442005
Train Epoch: 7 | Batch Status: 42000/50000 (84%) | Loss: 1.407252
Train Epoch: 7 | Batch Status: 42500/50000 (85%) | Loss: 1.250213
Train Epoch: 7 | Batch Status: 43000/50000 (86%) | Loss: 1.260042
Train Epoch: 7 | Batch Status: 43500/50000 (87%) | Loss: 1.221074
Train Epoch: 7 | Batch Status: 44000/50000 (88%) | Loss: 1.519031
Train Epoch: 7 | Batch Status: 44500/50000 (89%) | Loss: 1.486095
Train Epoch: 7 | Batch Status: 45000/50000 (90%) | Loss: 1.363160
Train Epoch: 7 | Batch Status: 45500/50000 (91%) | Loss: 1.461794
Train Epoch: 7 | Batch Status: 46000/50000 (92%) | Loss: 1.513821
Train Epoch: 7 | Batch Status: 46500/50000 (93%) | Loss: 1.409723
Train Epoch: 7 | Batch Status: 47000/50000 (94%) | Loss: 1.399649
Train Epoch: 7 | Batch Status: 47500/50000 (95%) | Loss: 1.600223
Train Epoch: 7 | Batch Status: 48000/50000 (96%) | Loss: 1.619546
Train Epoch: 7 | Batch Status: 48500/50000 (97%) | Loss: 1.351839
Train Epoch: 7 | Batch Status: 49000/50000 (98%) | Loss: 1.349633
Train Epoch: 7 | Batch Status: 49500/50000 (99%) | Loss: 1.278544
===========================
Test set: Average loss: 0.0287, Accuracy: 4833/10000 (48%)
Train Epoch: 8 | Batch Status: 0/50000 (0%) | Loss: 1.456976
Train Epoch: 8 | Batch Status: 500/50000 (1%) | Loss: 1.257439
Train Epoch: 8 | Batch Status: 1000/50000 (2%) | Loss: 1.460876
Train Epoch: 8 | Batch Status: 1500/50000 (3%) | Loss: 1.375468
Train Epoch: 8 | Batch Status: 2000/50000 (4%) | Loss: 1.188220
Train Epoch: 8 | Batch Status: 2500/50000 (5%) | Loss: 1.511474
Train Epoch: 8 | Batch Status: 3000/50000 (6%) | Loss: 1.202458
Train Epoch: 8 | Batch Status: 3500/50000 (7%) | Loss: 1.376578
Train Epoch: 8 | Batch Status: 4000/50000 (8%) | Loss: 1.359424
Train Epoch: 8 | Batch Status: 4500/50000 (9%) | Loss: 1.514946
Train Epoch: 8 | Batch Status: 5000/50000 (10%) | Loss: 1.550284
Train Epoch: 8 | Batch Status: 5500/50000 (11%) | Loss: 1.297421
Train Epoch: 8 | Batch Status: 6000/50000 (12%) | Loss: 1.276751
Train Epoch: 8 | Batch Status: 6500/50000 (13%) | Loss: 1.328430
Train Epoch: 8 | Batch Status: 7000/50000 (14%) | Loss: 1.166845
Train Epoch: 8 | Batch Status: 7500/50000 (15%) | Loss: 1.404267
Train Epoch: 8 | Batch Status: 8000/50000 (16%) | Loss: 1.654341
Train Epoch: 8 | Batch Status: 8500/50000 (17%) | Loss: 1.334022
Train Epoch: 8 | Batch Status: 9000/50000 (18%) | Loss: 1.511605
Train Epoch: 8 | Batch Status: 9500/50000 (19%) | Loss: 1.568304
Train Epoch: 8 | Batch Status: 10000/50000 (20%) | Loss: 1.088266
Train Epoch: 8 | Batch Status: 10500/50000 (21%) | Loss: 1.118620
Train Epoch: 8 | Batch Status: 11000/50000 (22%) | Loss: 1.212509
Train Epoch: 8 | Batch Status: 11500/50000 (23%) | Loss: 1.241494
Train Epoch: 8 | Batch Status: 12000/50000 (24%) | Loss: 1.430458
Train Epoch: 8 | Batch Status: 12500/50000 (25%) | Loss: 1.178420
Train Epoch: 8 | Batch Status: 13000/50000 (26%) | Loss: 1.345826
Train Epoch: 8 | Batch Status: 13500/50000 (27%) | Loss: 1.400575
Train Epoch: 8 | Batch Status: 14000/50000 (28%) | Loss: 1.782675
Train Epoch: 8 | Batch Status: 14500/50000 (29%) | Loss: 1.418155
Train Epoch: 8 | Batch Status: 15000/50000 (30%) | Loss: 1.357086
Train Epoch: 8 | Batch Status: 15500/50000 (31%) | Loss: 1.237534
Train Epoch: 8 | Batch Status: 16000/50000 (32%) | Loss: 1.346174
Train Epoch: 8 | Batch Status: 16500/50000 (33%) | Loss: 1.592406
Train Epoch: 8 | Batch Status: 17000/50000 (34%) | Loss: 1.598777
Train Epoch: 8 | Batch Status: 17500/50000 (35%) | Loss: 1.461269
Train Epoch: 8 | Batch Status: 18000/50000 (36%) | Loss: 1.399350
Train Epoch: 8 | Batch Status: 18500/50000 (37%) | Loss: 1.454071
Train Epoch: 8 | Batch Status: 19000/50000 (38%) | Loss: 1.335990
Train Epoch: 8 | Batch Status: 19500/50000 (39%) | Loss: 1.437259
Train Epoch: 8 | Batch Status: 20000/50000 (40%) | Loss: 1.365927
Train Epoch: 8 | Batch Status: 20500/50000 (41%) | Loss: 1.291335
Train Epoch: 8 | Batch Status: 21000/50000 (42%) | Loss: 1.163495
Train Epoch: 8 | Batch Status: 21500/50000 (43%) | Loss: 1.461856
Train Epoch: 8 | Batch Status: 22000/50000 (44%) | Loss: 1.364023
Train Epoch: 8 | Batch Status: 22500/50000 (45%) | Loss: 1.250683
Train Epoch: 8 | Batch Status: 23000/50000 (46%) | Loss: 1.268589
Train Epoch: 8 | Batch Status: 23500/50000 (47%) | Loss: 1.194443
Train Epoch: 8 | Batch Status: 24000/50000 (48%) | Loss: 1.329389
Train Epoch: 8 | Batch Status: 24500/50000 (49%) | Loss: 1.474723
Train Epoch: 8 | Batch Status: 25000/50000 (50%) | Loss: 1.241603
Train Epoch: 8 | Batch Status: 25500/50000 (51%) | Loss: 1.560221
Train Epoch: 8 | Batch Status: 26000/50000 (52%) | Loss: 1.059137
Train Epoch: 8 | Batch Status: 26500/50000 (53%) | Loss: 1.258641
Train Epoch: 8 | Batch Status: 27000/50000 (54%) | Loss: 1.352540
Train Epoch: 8 | Batch Status: 27500/50000 (55%) | Loss: 1.358048
Train Epoch: 8 | Batch Status: 28000/50000 (56%) | Loss: 1.227389
Train Epoch: 8 | Batch Status: 28500/50000 (57%) | Loss: 1.574095
Train Epoch: 8 | Batch Status: 29000/50000 (58%) | Loss: 1.328037
Train Epoch: 8 | Batch Status: 29500/50000 (59%) | Loss: 1.248037
Train Epoch: 8 | Batch Status: 30000/50000 (60%) | Loss: 1.451913
Train Epoch: 8 | Batch Status: 30500/50000 (61%) | Loss: 1.277421
Train Epoch: 8 | Batch Status: 31000/50000 (62%) | Loss: 1.474219
Train Epoch: 8 | Batch Status: 31500/50000 (63%) | Loss: 1.580326
Train Epoch: 8 | Batch Status: 32000/50000 (64%) | Loss: 1.281409
Train Epoch: 8 | Batch Status: 32500/50000 (65%) | Loss: 1.554186
Train Epoch: 8 | Batch Status: 33000/50000 (66%) | Loss: 1.293374
Train Epoch: 8 | Batch Status: 33500/50000 (67%) | Loss: 1.218721
Train Epoch: 8 | Batch Status: 34000/50000 (68%) | Loss: 1.174940
Train Epoch: 8 | Batch Status: 34500/50000 (69%) | Loss: 1.363631
Train Epoch: 8 | Batch Status: 35000/50000 (70%) | Loss: 1.262809
Train Epoch: 8 | Batch Status: 35500/50000 (71%) | Loss: 1.350429
Train Epoch: 8 | Batch Status: 36000/50000 (72%) | Loss: 1.355546
Train Epoch: 8 | Batch Status: 36500/50000 (73%) | Loss: 1.571415
Train Epoch: 8 | Batch Status: 37000/50000 (74%) | Loss: 1.289064
Train Epoch: 8 | Batch Status: 37500/50000 (75%) | Loss: 1.332802
Train Epoch: 8 | Batch Status: 38000/50000 (76%) | Loss: 1.517138
Train Epoch: 8 | Batch Status: 38500/50000 (77%) | Loss: 1.280701
Train Epoch: 8 | Batch Status: 39000/50000 (78%) | Loss: 1.398873
Train Epoch: 8 | Batch Status: 39500/50000 (79%) | Loss: 1.370435
Train Epoch: 8 | Batch Status: 40000/50000 (80%) | Loss: 1.291411
Train Epoch: 8 | Batch Status: 40500/50000 (81%) | Loss: 1.410932
Train Epoch: 8 | Batch Status: 41000/50000 (82%) | Loss: 1.123116
Train Epoch: 8 | Batch Status: 41500/50000 (83%) | Loss: 1.132648
Train Epoch: 8 | Batch Status: 42000/50000 (84%) | Loss: 1.618317
Train Epoch: 8 | Batch Status: 42500/50000 (85%) | Loss: 1.375842
Train Epoch: 8 | Batch Status: 43000/50000 (86%) | Loss: 1.534218
Train Epoch: 8 | Batch Status: 43500/50000 (87%) | Loss: 1.081820
Train Epoch: 8 | Batch Status: 44000/50000 (88%) | Loss: 1.148196
Train Epoch: 8 | Batch Status: 44500/50000 (89%) | Loss: 1.402401
Train Epoch: 8 | Batch Status: 45000/50000 (90%) | Loss: 1.350908
Train Epoch: 8 | Batch Status: 45500/50000 (91%) | Loss: 1.350984
Train Epoch: 8 | Batch Status: 46000/50000 (92%) | Loss: 1.360226
Train Epoch: 8 | Batch Status: 46500/50000 (93%) | Loss: 1.154029
Train Epoch: 8 | Batch Status: 47000/50000 (94%) | Loss: 1.383461
Train Epoch: 8 | Batch Status: 47500/50000 (95%) | Loss: 1.446115
Train Epoch: 8 | Batch Status: 48000/50000 (96%) | Loss: 1.305593
Train Epoch: 8 | Batch Status: 48500/50000 (97%) | Loss: 1.419236
Train Epoch: 8 | Batch Status: 49000/50000 (98%) | Loss: 1.393920
Train Epoch: 8 | Batch Status: 49500/50000 (99%) | Loss: 1.457589
===========================
Test set: Average loss: 0.0282, Accuracy: 4918/10000 (49%)
Train Epoch: 9 | Batch Status: 0/50000 (0%) | Loss: 1.248781
Train Epoch: 9 | Batch Status: 500/50000 (1%) | Loss: 1.316082
Train Epoch: 9 | Batch Status: 1000/50000 (2%) | Loss: 1.706764
Train Epoch: 9 | Batch Status: 1500/50000 (3%) | Loss: 1.166518
Train Epoch: 9 | Batch Status: 2000/50000 (4%) | Loss: 1.199667
Train Epoch: 9 | Batch Status: 2500/50000 (5%) | Loss: 1.571233
Train Epoch: 9 | Batch Status: 3000/50000 (6%) | Loss: 1.467836
Train Epoch: 9 | Batch Status: 3500/50000 (7%) | Loss: 1.560943
Train Epoch: 9 | Batch Status: 4000/50000 (8%) | Loss: 1.390063
Train Epoch: 9 | Batch Status: 4500/50000 (9%) | Loss: 1.476633
Train Epoch: 9 | Batch Status: 5000/50000 (10%) | Loss: 1.214643
Train Epoch: 9 | Batch Status: 5500/50000 (11%) | Loss: 1.316957
Train Epoch: 9 | Batch Status: 6000/50000 (12%) | Loss: 1.417546
Train Epoch: 9 | Batch Status: 6500/50000 (13%) | Loss: 1.276018
Train Epoch: 9 | Batch Status: 7000/50000 (14%) | Loss: 1.077275
Train Epoch: 9 | Batch Status: 7500/50000 (15%) | Loss: 1.561195
Train Epoch: 9 | Batch Status: 8000/50000 (16%) | Loss: 1.426885
Train Epoch: 9 | Batch Status: 8500/50000 (17%) | Loss: 1.362332
Train Epoch: 9 | Batch Status: 9000/50000 (18%) | Loss: 1.486863
Train Epoch: 9 | Batch Status: 9500/50000 (19%) | Loss: 1.340747
Train Epoch: 9 | Batch Status: 10000/50000 (20%) | Loss: 1.198512
Train Epoch: 9 | Batch Status: 10500/50000 (21%) | Loss: 1.475580
Train Epoch: 9 | Batch Status: 11000/50000 (22%) | Loss: 1.452271
Train Epoch: 9 | Batch Status: 11500/50000 (23%) | Loss: 1.160216
Train Epoch: 9 | Batch Status: 12000/50000 (24%) | Loss: 1.265468
Train Epoch: 9 | Batch Status: 12500/50000 (25%) | Loss: 1.812719
Train Epoch: 9 | Batch Status: 13000/50000 (26%) | Loss: 1.472034
Train Epoch: 9 | Batch Status: 13500/50000 (27%) | Loss: 1.401290
Train Epoch: 9 | Batch Status: 14000/50000 (28%) | Loss: 1.445383
Train Epoch: 9 | Batch Status: 14500/50000 (29%) | Loss: 1.582448
Train Epoch: 9 | Batch Status: 15000/50000 (30%) | Loss: 1.368682
Train Epoch: 9 | Batch Status: 15500/50000 (31%) | Loss: 1.442402
Train Epoch: 9 | Batch Status: 16000/50000 (32%) | Loss: 1.415573
Train Epoch: 9 | Batch Status: 16500/50000 (33%) | Loss: 1.489617
Train Epoch: 9 | Batch Status: 17000/50000 (34%) | Loss: 1.300099
Train Epoch: 9 | Batch Status: 17500/50000 (35%) | Loss: 1.342408
Train Epoch: 9 | Batch Status: 18000/50000 (36%) | Loss: 1.218114
Train Epoch: 9 | Batch Status: 18500/50000 (37%) | Loss: 1.452853
Train Epoch: 9 | Batch Status: 19000/50000 (38%) | Loss: 1.279144
Train Epoch: 9 | Batch Status: 19500/50000 (39%) | Loss: 1.323450
Train Epoch: 9 | Batch Status: 20000/50000 (40%) | Loss: 1.297816
Train Epoch: 9 | Batch Status: 20500/50000 (41%) | Loss: 1.315930
Train Epoch: 9 | Batch Status: 21000/50000 (42%) | Loss: 1.205885
Train Epoch: 9 | Batch Status: 21500/50000 (43%) | Loss: 1.396127
Train Epoch: 9 | Batch Status: 22000/50000 (44%) | Loss: 1.227839
Train Epoch: 9 | Batch Status: 22500/50000 (45%) | Loss: 1.104246
Train Epoch: 9 | Batch Status: 23000/50000 (46%) | Loss: 1.252795
Train Epoch: 9 | Batch Status: 23500/50000 (47%) | Loss: 1.368808
Train Epoch: 9 | Batch Status: 24000/50000 (48%) | Loss: 1.187183
Train Epoch: 9 | Batch Status: 24500/50000 (49%) | Loss: 1.292181
Train Epoch: 9 | Batch Status: 25000/50000 (50%) | Loss: 1.686219
Train Epoch: 9 | Batch Status: 25500/50000 (51%) | Loss: 1.121328
Train Epoch: 9 | Batch Status: 26000/50000 (52%) | Loss: 1.600171
Train Epoch: 9 | Batch Status: 26500/50000 (53%) | Loss: 1.324311
Train Epoch: 9 | Batch Status: 27000/50000 (54%) | Loss: 1.343912
Train Epoch: 9 | Batch Status: 27500/50000 (55%) | Loss: 1.145477
Train Epoch: 9 | Batch Status: 28000/50000 (56%) | Loss: 1.579499
Train Epoch: 9 | Batch Status: 28500/50000 (57%) | Loss: 1.529229
Train Epoch: 9 | Batch Status: 29000/50000 (58%) | Loss: 1.068586
Train Epoch: 9 | Batch Status: 29500/50000 (59%) | Loss: 1.290940
Train Epoch: 9 | Batch Status: 30000/50000 (60%) | Loss: 1.452065
Train Epoch: 9 | Batch Status: 30500/50000 (61%) | Loss: 1.260037
Train Epoch: 9 | Batch Status: 31000/50000 (62%) | Loss: 1.388105
Train Epoch: 9 | Batch Status: 31500/50000 (63%) | Loss: 1.204632
Train Epoch: 9 | Batch Status: 32000/50000 (64%) | Loss: 1.375756
Train Epoch: 9 | Batch Status: 32500/50000 (65%) | Loss: 1.495874
Train Epoch: 9 | Batch Status: 33000/50000 (66%) | Loss: 1.498058
Train Epoch: 9 | Batch Status: 33500/50000 (67%) | Loss: 1.263392
Train Epoch: 9 | Batch Status: 34000/50000 (68%) | Loss: 1.209887
Train Epoch: 9 | Batch Status: 34500/50000 (69%) | Loss: 1.132780
Train Epoch: 9 | Batch Status: 35000/50000 (70%) | Loss: 1.237662
Train Epoch: 9 | Batch Status: 35500/50000 (71%) | Loss: 1.508318
Train Epoch: 9 | Batch Status: 36000/50000 (72%) | Loss: 1.246070
Train Epoch: 9 | Batch Status: 36500/50000 (73%) | Loss: 1.059466
Train Epoch: 9 | Batch Status: 37000/50000 (74%) | Loss: 1.204705
Train Epoch: 9 | Batch Status: 37500/50000 (75%) | Loss: 1.118329
Train Epoch: 9 | Batch Status: 38000/50000 (76%) | Loss: 1.096867
Train Epoch: 9 | Batch Status: 38500/50000 (77%) | Loss: 1.336452
Train Epoch: 9 | Batch Status: 39000/50000 (78%) | Loss: 1.366925
Train Epoch: 9 | Batch Status: 39500/50000 (79%) | Loss: 1.540546
Train Epoch: 9 | Batch Status: 40000/50000 (80%) | Loss: 1.380799
Train Epoch: 9 | Batch Status: 40500/50000 (81%) | Loss: 1.461064
Train Epoch: 9 | Batch Status: 41000/50000 (82%) | Loss: 1.234603
Train Epoch: 9 | Batch Status: 41500/50000 (83%) | Loss: 1.244372
Train Epoch: 9 | Batch Status: 42000/50000 (84%) | Loss: 1.212512
Train Epoch: 9 | Batch Status: 42500/50000 (85%) | Loss: 1.154936
Train Epoch: 9 | Batch Status: 43000/50000 (86%) | Loss: 1.192320
Train Epoch: 9 | Batch Status: 43500/50000 (87%) | Loss: 1.327924
Train Epoch: 9 | Batch Status: 44000/50000 (88%) | Loss: 1.075498
Train Epoch: 9 | Batch Status: 44500/50000 (89%) | Loss: 1.240917
Train Epoch: 9 | Batch Status: 45000/50000 (90%) | Loss: 1.340158
Train Epoch: 9 | Batch Status: 45500/50000 (91%) | Loss: 1.407730
Train Epoch: 9 | Batch Status: 46000/50000 (92%) | Loss: 1.268955
Train Epoch: 9 | Batch Status: 46500/50000 (93%) | Loss: 1.475997
Train Epoch: 9 | Batch Status: 47000/50000 (94%) | Loss: 1.181730
Train Epoch: 9 | Batch Status: 47500/50000 (95%) | Loss: 1.469310
Train Epoch: 9 | Batch Status: 48000/50000 (96%) | Loss: 1.122650
Train Epoch: 9 | Batch Status: 48500/50000 (97%) | Loss: 1.490864
Train Epoch: 9 | Batch Status: 49000/50000 (98%) | Loss: 1.351797
Train Epoch: 9 | Batch Status: 49500/50000 (99%) | Loss: 1.210996
===========================
Test set: Average loss: 0.0294, Accuracy: 4967/10000 (50%)
Train Epoch: 10 | Batch Status: 0/50000 (0%) | Loss: 1.322497
Train Epoch: 10 | Batch Status: 500/50000 (1%) | Loss: 1.195039
Train Epoch: 10 | Batch Status: 1000/50000 (2%) | Loss: 1.188042
Train Epoch: 10 | Batch Status: 1500/50000 (3%) | Loss: 1.427402
Train Epoch: 10 | Batch Status: 2000/50000 (4%) | Loss: 1.144553
Train Epoch: 10 | Batch Status: 2500/50000 (5%) | Loss: 1.249363
Train Epoch: 10 | Batch Status: 3000/50000 (6%) | Loss: 1.219796
Train Epoch: 10 | Batch Status: 3500/50000 (7%) | Loss: 1.173643
Train Epoch: 10 | Batch Status: 4000/50000 (8%) | Loss: 1.403092
Train Epoch: 10 | Batch Status: 4500/50000 (9%) | Loss: 1.437947
Train Epoch: 10 | Batch Status: 5000/50000 (10%) | Loss: 1.027186
Train Epoch: 10 | Batch Status: 5500/50000 (11%) | Loss: 1.373358
Train Epoch: 10 | Batch Status: 6000/50000 (12%) | Loss: 1.002303
Train Epoch: 10 | Batch Status: 6500/50000 (13%) | Loss: 1.484800
Train Epoch: 10 | Batch Status: 7000/50000 (14%) | Loss: 1.139847
Train Epoch: 10 | Batch Status: 7500/50000 (15%) | Loss: 1.091776
Train Epoch: 10 | Batch Status: 8000/50000 (16%) | Loss: 1.192107
Train Epoch: 10 | Batch Status: 8500/50000 (17%) | Loss: 1.270904
Train Epoch: 10 | Batch Status: 9000/50000 (18%) | Loss: 1.435087
Train Epoch: 10 | Batch Status: 9500/50000 (19%) | Loss: 1.545702
Train Epoch: 10 | Batch Status: 10000/50000 (20%) | Loss: 1.152311
Train Epoch: 10 | Batch Status: 10500/50000 (21%) | Loss: 1.065000
Train Epoch: 10 | Batch Status: 11000/50000 (22%) | Loss: 1.075917
Train Epoch: 10 | Batch Status: 11500/50000 (23%) | Loss: 1.307229
Train Epoch: 10 | Batch Status: 12000/50000 (24%) | Loss: 1.168905
Train Epoch: 10 | Batch Status: 12500/50000 (25%) | Loss: 1.410222
Train Epoch: 10 | Batch Status: 13000/50000 (26%) | Loss: 1.351879
Train Epoch: 10 | Batch Status: 13500/50000 (27%) | Loss: 1.658927
Train Epoch: 10 | Batch Status: 14000/50000 (28%) | Loss: 1.153459
Train Epoch: 10 | Batch Status: 14500/50000 (29%) | Loss: 1.159621
Train Epoch: 10 | Batch Status: 15000/50000 (30%) | Loss: 1.272752
Train Epoch: 10 | Batch Status: 15500/50000 (31%) | Loss: 1.258271
Train Epoch: 10 | Batch Status: 16000/50000 (32%) | Loss: 1.300213
Train Epoch: 10 | Batch Status: 16500/50000 (33%) | Loss: 1.634277
Train Epoch: 10 | Batch Status: 17000/50000 (34%) | Loss: 1.538037
Train Epoch: 10 | Batch Status: 17500/50000 (35%) | Loss: 1.295508
Train Epoch: 10 | Batch Status: 18000/50000 (36%) | Loss: 1.226388
Train Epoch: 10 | Batch Status: 18500/50000 (37%) | Loss: 1.458466
Train Epoch: 10 | Batch Status: 19000/50000 (38%) | Loss: 1.161494
Train Epoch: 10 | Batch Status: 19500/50000 (39%) | Loss: 1.537049
Train Epoch: 10 | Batch Status: 20000/50000 (40%) | Loss: 1.313890
Train Epoch: 10 | Batch Status: 20500/50000 (41%) | Loss: 1.278410
Train Epoch: 10 | Batch Status: 21000/50000 (42%) | Loss: 1.200115
Train Epoch: 10 | Batch Status: 21500/50000 (43%) | Loss: 1.227400
Train Epoch: 10 | Batch Status: 22000/50000 (44%) | Loss: 1.135542
Train Epoch: 10 | Batch Status: 22500/50000 (45%) | Loss: 1.290350
Train Epoch: 10 | Batch Status: 23000/50000 (46%) | Loss: 0.994189
Train Epoch: 10 | Batch Status: 23500/50000 (47%) | Loss: 1.264415
Train Epoch: 10 | Batch Status: 24000/50000 (48%) | Loss: 1.407567
Train Epoch: 10 | Batch Status: 24500/50000 (49%) | Loss: 1.267060
Train Epoch: 10 | Batch Status: 25000/50000 (50%) | Loss: 0.935436
Train Epoch: 10 | Batch Status: 25500/50000 (51%) | Loss: 1.121220
Train Epoch: 10 | Batch Status: 26000/50000 (52%) | Loss: 1.338609
Train Epoch: 10 | Batch Status: 26500/50000 (53%) | Loss: 1.319948
Train Epoch: 10 | Batch Status: 27000/50000 (54%) | Loss: 1.393894
Train Epoch: 10 | Batch Status: 27500/50000 (55%) | Loss: 1.267953
Train Epoch: 10 | Batch Status: 28000/50000 (56%) | Loss: 1.440347
Train Epoch: 10 | Batch Status: 28500/50000 (57%) | Loss: 1.211135
Train Epoch: 10 | Batch Status: 29000/50000 (58%) | Loss: 1.206592
Train Epoch: 10 | Batch Status: 29500/50000 (59%) | Loss: 1.369069
Train Epoch: 10 | Batch Status: 30000/50000 (60%) | Loss: 1.353441
Train Epoch: 10 | Batch Status: 30500/50000 (61%) | Loss: 1.141687
Train Epoch: 10 | Batch Status: 31000/50000 (62%) | Loss: 1.143528
Train Epoch: 10 | Batch Status: 31500/50000 (63%) | Loss: 1.260763
Train Epoch: 10 | Batch Status: 32000/50000 (64%) | Loss: 0.968847
Train Epoch: 10 | Batch Status: 32500/50000 (65%) | Loss: 1.380209
Train Epoch: 10 | Batch Status: 33000/50000 (66%) | Loss: 1.345861
Train Epoch: 10 | Batch Status: 33500/50000 (67%) | Loss: 1.138566
Train Epoch: 10 | Batch Status: 34000/50000 (68%) | Loss: 1.573477
Train Epoch: 10 | Batch Status: 34500/50000 (69%) | Loss: 1.519846
Train Epoch: 10 | Batch Status: 35000/50000 (70%) | Loss: 1.152257
Train Epoch: 10 | Batch Status: 35500/50000 (71%) | Loss: 0.961616
Train Epoch: 10 | Batch Status: 36000/50000 (72%) | Loss: 1.530760
Train Epoch: 10 | Batch Status: 36500/50000 (73%) | Loss: 1.216147
Train Epoch: 10 | Batch Status: 37000/50000 (74%) | Loss: 1.179148
Train Epoch: 10 | Batch Status: 37500/50000 (75%) | Loss: 1.229755
Train Epoch: 10 | Batch Status: 38000/50000 (76%) | Loss: 1.304945
Train Epoch: 10 | Batch Status: 38500/50000 (77%) | Loss: 1.182360
Train Epoch: 10 | Batch Status: 39000/50000 (78%) | Loss: 1.054660
Train Epoch: 10 | Batch Status: 39500/50000 (79%) | Loss: 1.217547
Train Epoch: 10 | Batch Status: 40000/50000 (80%) | Loss: 1.014256
Train Epoch: 10 | Batch Status: 40500/50000 (81%) | Loss: 1.099281
Train Epoch: 10 | Batch Status: 41000/50000 (82%) | Loss: 1.402900
Train Epoch: 10 | Batch Status: 41500/50000 (83%) | Loss: 1.501342
Train Epoch: 10 | Batch Status: 42000/50000 (84%) | Loss: 1.561828
Train Epoch: 10 | Batch Status: 42500/50000 (85%) | Loss: 0.964529
Train Epoch: 10 | Batch Status: 43000/50000 (86%) | Loss: 1.256698
Train Epoch: 10 | Batch Status: 43500/50000 (87%) | Loss: 1.161736
Train Epoch: 10 | Batch Status: 44000/50000 (88%) | Loss: 1.203791
Train Epoch: 10 | Batch Status: 44500/50000 (89%) | Loss: 1.331448
Train Epoch: 10 | Batch Status: 45000/50000 (90%) | Loss: 1.183346
Train Epoch: 10 | Batch Status: 45500/50000 (91%) | Loss: 1.181661
Train Epoch: 10 | Batch Status: 46000/50000 (92%) | Loss: 1.074583
Train Epoch: 10 | Batch Status: 46500/50000 (93%) | Loss: 1.507713
Train Epoch: 10 | Batch Status: 47000/50000 (94%) | Loss: 1.280044
Train Epoch: 10 | Batch Status: 47500/50000 (95%) | Loss: 1.409165
Train Epoch: 10 | Batch Status: 48000/50000 (96%) | Loss: 1.295431
Train Epoch: 10 | Batch Status: 48500/50000 (97%) | Loss: 1.281298
Train Epoch: 10 | Batch Status: 49000/50000 (98%) | Loss: 1.124535
Train Epoch: 10 | Batch Status: 49500/50000 (99%) | Loss: 1.465933
===========================
Test set: Average loss: 0.0276, Accuracy: 5089/10000 (51%)
Train Epoch: 11 | Batch Status: 0/50000 (0%) | Loss: 1.176187
Train Epoch: 11 | Batch Status: 500/50000 (1%) | Loss: 1.162594
Train Epoch: 11 | Batch Status: 1000/50000 (2%) | Loss: 1.286005
Train Epoch: 11 | Batch Status: 1500/50000 (3%) | Loss: 1.173018
Train Epoch: 11 | Batch Status: 2000/50000 (4%) | Loss: 1.184286
Train Epoch: 11 | Batch Status: 2500/50000 (5%) | Loss: 1.149880
Train Epoch: 11 | Batch Status: 3000/50000 (6%) | Loss: 1.025348
Train Epoch: 11 | Batch Status: 3500/50000 (7%) | Loss: 1.450871
Train Epoch: 11 | Batch Status: 4000/50000 (8%) | Loss: 1.516531
Train Epoch: 11 | Batch Status: 4500/50000 (9%) | Loss: 1.278853
Train Epoch: 11 | Batch Status: 5000/50000 (10%) | Loss: 1.224625
Train Epoch: 11 | Batch Status: 5500/50000 (11%) | Loss: 1.109595
Train Epoch: 11 | Batch Status: 6000/50000 (12%) | Loss: 1.253284
Train Epoch: 11 | Batch Status: 6500/50000 (13%) | Loss: 1.291793
Train Epoch: 11 | Batch Status: 7000/50000 (14%) | Loss: 1.413828
Train Epoch: 11 | Batch Status: 7500/50000 (15%) | Loss: 1.118700
Train Epoch: 11 | Batch Status: 8000/50000 (16%) | Loss: 1.228547
Train Epoch: 11 | Batch Status: 8500/50000 (17%) | Loss: 1.469539
Train Epoch: 11 | Batch Status: 9000/50000 (18%) | Loss: 1.267938
Train Epoch: 11 | Batch Status: 9500/50000 (19%) | Loss: 1.339864
Train Epoch: 11 | Batch Status: 10000/50000 (20%) | Loss: 1.270675
Train Epoch: 11 | Batch Status: 10500/50000 (21%) | Loss: 1.062488
Train Epoch: 11 | Batch Status: 11000/50000 (22%) | Loss: 1.176116
Train Epoch: 11 | Batch Status: 11500/50000 (23%) | Loss: 1.145655
Train Epoch: 11 | Batch Status: 12000/50000 (24%) | Loss: 1.291265
Train Epoch: 11 | Batch Status: 12500/50000 (25%) | Loss: 1.319654
Train Epoch: 11 | Batch Status: 13000/50000 (26%) | Loss: 1.087519
Train Epoch: 11 | Batch Status: 13500/50000 (27%) | Loss: 1.442848
Train Epoch: 11 | Batch Status: 14000/50000 (28%) | Loss: 1.120350
Train Epoch: 11 | Batch Status: 14500/50000 (29%) | Loss: 0.962134
Train Epoch: 11 | Batch Status: 15000/50000 (30%) | Loss: 1.158026
Train Epoch: 11 | Batch Status: 15500/50000 (31%) | Loss: 0.865084
Train Epoch: 11 | Batch Status: 16000/50000 (32%) | Loss: 1.172481
Train Epoch: 11 | Batch Status: 16500/50000 (33%) | Loss: 1.017007
Train Epoch: 11 | Batch Status: 17000/50000 (34%) | Loss: 1.316009
Train Epoch: 11 | Batch Status: 17500/50000 (35%) | Loss: 1.475931
Train Epoch: 11 | Batch Status: 18000/50000 (36%) | Loss: 1.269966
Train Epoch: 11 | Batch Status: 18500/50000 (37%) | Loss: 1.166208
Train Epoch: 11 | Batch Status: 19000/50000 (38%) | Loss: 1.134583
Train Epoch: 11 | Batch Status: 19500/50000 (39%) | Loss: 1.125532
Train Epoch: 11 | Batch Status: 20000/50000 (40%) | Loss: 1.218500
Train Epoch: 11 | Batch Status: 20500/50000 (41%) | Loss: 1.203545
Train Epoch: 11 | Batch Status: 21000/50000 (42%) | Loss: 1.269791
Train Epoch: 11 | Batch Status: 21500/50000 (43%) | Loss: 1.079635
Train Epoch: 11 | Batch Status: 22000/50000 (44%) | Loss: 1.081472
Train Epoch: 11 | Batch Status: 22500/50000 (45%) | Loss: 1.197746
Train Epoch: 11 | Batch Status: 23000/50000 (46%) | Loss: 1.419473
Train Epoch: 11 | Batch Status: 23500/50000 (47%) | Loss: 1.237968
Train Epoch: 11 | Batch Status: 24000/50000 (48%) | Loss: 1.383629
Train Epoch: 11 | Batch Status: 24500/50000 (49%) | Loss: 1.178876
Train Epoch: 11 | Batch Status: 25000/50000 (50%) | Loss: 1.176269
Train Epoch: 11 | Batch Status: 25500/50000 (51%) | Loss: 1.268653
Train Epoch: 11 | Batch Status: 26000/50000 (52%) | Loss: 1.045271
Train Epoch: 11 | Batch Status: 26500/50000 (53%) | Loss: 1.408081
Train Epoch: 11 | Batch Status: 27000/50000 (54%) | Loss: 1.411890
Train Epoch: 11 | Batch Status: 27500/50000 (55%) | Loss: 1.295994
Train Epoch: 11 | Batch Status: 28000/50000 (56%) | Loss: 1.464856
Train Epoch: 11 | Batch Status: 28500/50000 (57%) | Loss: 1.214954
Train Epoch: 11 | Batch Status: 29000/50000 (58%) | Loss: 1.166598
Train Epoch: 11 | Batch Status: 29500/50000 (59%) | Loss: 1.161529
Train Epoch: 11 | Batch Status: 30000/50000 (60%) | Loss: 1.052637
Train Epoch: 11 | Batch Status: 30500/50000 (61%) | Loss: 1.393381
Train Epoch: 11 | Batch Status: 31000/50000 (62%) | Loss: 1.035549
Train Epoch: 11 | Batch Status: 31500/50000 (63%) | Loss: 1.395998
Train Epoch: 11 | Batch Status: 32000/50000 (64%) | Loss: 1.154427
Train Epoch: 11 | Batch Status: 32500/50000 (65%) | Loss: 1.335815
Train Epoch: 11 | Batch Status: 33000/50000 (66%) | Loss: 1.095158
Train Epoch: 11 | Batch Status: 33500/50000 (67%) | Loss: 1.036664
Train Epoch: 11 | Batch Status: 34000/50000 (68%) | Loss: 1.045790
Train Epoch: 11 | Batch Status: 34500/50000 (69%) | Loss: 1.248089
Train Epoch: 11 | Batch Status: 35000/50000 (70%) | Loss: 1.620047
Train Epoch: 11 | Batch Status: 35500/50000 (71%) | Loss: 1.165923
Train Epoch: 11 | Batch Status: 36000/50000 (72%) | Loss: 1.074252
Train Epoch: 11 | Batch Status: 36500/50000 (73%) | Loss: 1.065769
Train Epoch: 11 | Batch Status: 37000/50000 (74%) | Loss: 1.262681
Train Epoch: 11 | Batch Status: 37500/50000 (75%) | Loss: 1.167092
Train Epoch: 11 | Batch Status: 38000/50000 (76%) | Loss: 1.283337
Train Epoch: 11 | Batch Status: 38500/50000 (77%) | Loss: 1.414126
Train Epoch: 11 | Batch Status: 39000/50000 (78%) | Loss: 1.264009
Train Epoch: 11 | Batch Status: 39500/50000 (79%) | Loss: 1.190835
Train Epoch: 11 | Batch Status: 40000/50000 (80%) | Loss: 1.289582
Train Epoch: 11 | Batch Status: 40500/50000 (81%) | Loss: 1.240233
Train Epoch: 11 | Batch Status: 41000/50000 (82%) | Loss: 1.200945
Train Epoch: 11 | Batch Status: 41500/50000 (83%) | Loss: 1.193716
Train Epoch: 11 | Batch Status: 42000/50000 (84%) | Loss: 1.298326
Train Epoch: 11 | Batch Status: 42500/50000 (85%) | Loss: 0.984194
Train Epoch: 11 | Batch Status: 43000/50000 (86%) | Loss: 1.085249
Train Epoch: 11 | Batch Status: 43500/50000 (87%) | Loss: 1.278822
Train Epoch: 11 | Batch Status: 44000/50000 (88%) | Loss: 1.275179
Train Epoch: 11 | Batch Status: 44500/50000 (89%) | Loss: 1.293617
Train Epoch: 11 | Batch Status: 45000/50000 (90%) | Loss: 1.054137
Train Epoch: 11 | Batch Status: 45500/50000 (91%) | Loss: 1.378955
Train Epoch: 11 | Batch Status: 46000/50000 (92%) | Loss: 1.224089
Train Epoch: 11 | Batch Status: 46500/50000 (93%) | Loss: 1.235675
Train Epoch: 11 | Batch Status: 47000/50000 (94%) | Loss: 1.466322
Train Epoch: 11 | Batch Status: 47500/50000 (95%) | Loss: 0.940478
Train Epoch: 11 | Batch Status: 48000/50000 (96%) | Loss: 1.374773
Train Epoch: 11 | Batch Status: 48500/50000 (97%) | Loss: 1.475379
Train Epoch: 11 | Batch Status: 49000/50000 (98%) | Loss: 1.100497
Train Epoch: 11 | Batch Status: 49500/50000 (99%) | Loss: 1.225903
===========================
Test set: Average loss: 0.0274, Accuracy: 5156/10000 (52%)
Train Epoch: 12 | Batch Status: 0/50000 (0%) | Loss: 1.067141
Train Epoch: 12 | Batch Status: 500/50000 (1%) | Loss: 1.078697
Train Epoch: 12 | Batch Status: 1000/50000 (2%) | Loss: 1.364910
Train Epoch: 12 | Batch Status: 1500/50000 (3%) | Loss: 0.912026
Train Epoch: 12 | Batch Status: 2000/50000 (4%) | Loss: 1.198679
Train Epoch: 12 | Batch Status: 2500/50000 (5%) | Loss: 1.186352
Train Epoch: 12 | Batch Status: 3000/50000 (6%) | Loss: 1.119905
Train Epoch: 12 | Batch Status: 3500/50000 (7%) | Loss: 1.265556
Train Epoch: 12 | Batch Status: 4000/50000 (8%) | Loss: 1.255109
Train Epoch: 12 | Batch Status: 4500/50000 (9%) | Loss: 1.009824
Train Epoch: 12 | Batch Status: 5000/50000 (10%) | Loss: 1.011283
Train Epoch: 12 | Batch Status: 5500/50000 (11%) | Loss: 1.148819
Train Epoch: 12 | Batch Status: 6000/50000 (12%) | Loss: 1.120608
Train Epoch: 12 | Batch Status: 6500/50000 (13%) | Loss: 0.963193
Train Epoch: 12 | Batch Status: 7000/50000 (14%) | Loss: 1.173513
Train Epoch: 12 | Batch Status: 7500/50000 (15%) | Loss: 1.213298
Train Epoch: 12 | Batch Status: 8000/50000 (16%) | Loss: 1.208311
Train Epoch: 12 | Batch Status: 8500/50000 (17%) | Loss: 1.047279
Train Epoch: 12 | Batch Status: 9000/50000 (18%) | Loss: 1.128794
Train Epoch: 12 | Batch Status: 9500/50000 (19%) | Loss: 0.951917
Train Epoch: 12 | Batch Status: 10000/50000 (20%) | Loss: 1.614025
Train Epoch: 12 | Batch Status: 10500/50000 (21%) | Loss: 0.994390
Train Epoch: 12 | Batch Status: 11000/50000 (22%) | Loss: 1.190586
Train Epoch: 12 | Batch Status: 11500/50000 (23%) | Loss: 1.085835
Train Epoch: 12 | Batch Status: 12000/50000 (24%) | Loss: 1.115334
Train Epoch: 12 | Batch Status: 12500/50000 (25%) | Loss: 1.051008
Train Epoch: 12 | Batch Status: 13000/50000 (26%) | Loss: 1.144088
Train Epoch: 12 | Batch Status: 13500/50000 (27%) | Loss: 1.264840
Train Epoch: 12 | Batch Status: 14000/50000 (28%) | Loss: 1.350904
Train Epoch: 12 | Batch Status: 14500/50000 (29%) | Loss: 1.022123
Train Epoch: 12 | Batch Status: 15000/50000 (30%) | Loss: 1.150474
Train Epoch: 12 | Batch Status: 15500/50000 (31%) | Loss: 1.331155
Train Epoch: 12 | Batch Status: 16000/50000 (32%) | Loss: 1.200510
Train Epoch: 12 | Batch Status: 16500/50000 (33%) | Loss: 1.028188
Train Epoch: 12 | Batch Status: 17000/50000 (34%) | Loss: 1.085084
Train Epoch: 12 | Batch Status: 17500/50000 (35%) | Loss: 1.431909
Train Epoch: 12 | Batch Status: 18000/50000 (36%) | Loss: 1.224457
Train Epoch: 12 | Batch Status: 18500/50000 (37%) | Loss: 1.048018
Train Epoch: 12 | Batch Status: 19000/50000 (38%) | Loss: 0.713465
Train Epoch: 12 | Batch Status: 19500/50000 (39%) | Loss: 1.222940
Train Epoch: 12 | Batch Status: 20000/50000 (40%) | Loss: 1.046437
Train Epoch: 12 | Batch Status: 20500/50000 (41%) | Loss: 1.063156
Train Epoch: 12 | Batch Status: 21000/50000 (42%) | Loss: 1.103021
Train Epoch: 12 | Batch Status: 21500/50000 (43%) | Loss: 1.187106
Train Epoch: 12 | Batch Status: 22000/50000 (44%) | Loss: 1.308082
Train Epoch: 12 | Batch Status: 22500/50000 (45%) | Loss: 1.029644
Train Epoch: 12 | Batch Status: 23000/50000 (46%) | Loss: 1.103475
Train Epoch: 12 | Batch Status: 23500/50000 (47%) | Loss: 1.262267
Train Epoch: 12 | Batch Status: 24000/50000 (48%) | Loss: 1.201432
Train Epoch: 12 | Batch Status: 24500/50000 (49%) | Loss: 1.410979
Train Epoch: 12 | Batch Status: 25000/50000 (50%) | Loss: 1.053360
Train Epoch: 12 | Batch Status: 25500/50000 (51%) | Loss: 1.300377
Train Epoch: 12 | Batch Status: 26000/50000 (52%) | Loss: 1.376494
Train Epoch: 12 | Batch Status: 26500/50000 (53%) | Loss: 1.028183
Train Epoch: 12 | Batch Status: 27000/50000 (54%) | Loss: 1.173234
Train Epoch: 12 | Batch Status: 27500/50000 (55%) | Loss: 1.178806
Train Epoch: 12 | Batch Status: 28000/50000 (56%) | Loss: 1.039570
Train Epoch: 12 | Batch Status: 28500/50000 (57%) | Loss: 1.339647
Train Epoch: 12 | Batch Status: 29000/50000 (58%) | Loss: 1.356018
Train Epoch: 12 | Batch Status: 29500/50000 (59%) | Loss: 1.194237
Train Epoch: 12 | Batch Status: 30000/50000 (60%) | Loss: 1.105590
Train Epoch: 12 | Batch Status: 30500/50000 (61%) | Loss: 1.221992
Train Epoch: 12 | Batch Status: 31000/50000 (62%) | Loss: 1.279023
Train Epoch: 12 | Batch Status: 31500/50000 (63%) | Loss: 1.448981
Train Epoch: 12 | Batch Status: 32000/50000 (64%) | Loss: 0.946459
Train Epoch: 12 | Batch Status: 32500/50000 (65%) | Loss: 1.188748
Train Epoch: 12 | Batch Status: 33000/50000 (66%) | Loss: 0.966137
Train Epoch: 12 | Batch Status: 33500/50000 (67%) | Loss: 1.067976
Train Epoch: 12 | Batch Status: 34000/50000 (68%) | Loss: 1.500881
Train Epoch: 12 | Batch Status: 34500/50000 (69%) | Loss: 1.334809
Train Epoch: 12 | Batch Status: 35000/50000 (70%) | Loss: 1.021206
Train Epoch: 12 | Batch Status: 35500/50000 (71%) | Loss: 1.171020
Train Epoch: 12 | Batch Status: 36000/50000 (72%) | Loss: 1.081272
Train Epoch: 12 | Batch Status: 36500/50000 (73%) | Loss: 1.263528
Train Epoch: 12 | Batch Status: 37000/50000 (74%) | Loss: 1.284703
Train Epoch: 12 | Batch Status: 37500/50000 (75%) | Loss: 1.159236
Train Epoch: 12 | Batch Status: 38000/50000 (76%) | Loss: 1.233639
Train Epoch: 12 | Batch Status: 38500/50000 (77%) | Loss: 1.316303
Train Epoch: 12 | Batch Status: 39000/50000 (78%) | Loss: 1.272363
Train Epoch: 12 | Batch Status: 39500/50000 (79%) | Loss: 1.162231
Train Epoch: 12 | Batch Status: 40000/50000 (80%) | Loss: 1.231526
Train Epoch: 12 | Batch Status: 40500/50000 (81%) | Loss: 1.245206
Train Epoch: 12 | Batch Status: 41000/50000 (82%) | Loss: 1.331909
Train Epoch: 12 | Batch Status: 41500/50000 (83%) | Loss: 1.535856
Train Epoch: 12 | Batch Status: 42000/50000 (84%) | Loss: 1.340201
Train Epoch: 12 | Batch Status: 42500/50000 (85%) | Loss: 1.322405
Train Epoch: 12 | Batch Status: 43000/50000 (86%) | Loss: 0.989708
Train Epoch: 12 | Batch Status: 43500/50000 (87%) | Loss: 1.105656
Train Epoch: 12 | Batch Status: 44000/50000 (88%) | Loss: 1.431957
Train Epoch: 12 | Batch Status: 44500/50000 (89%) | Loss: 1.193018
Train Epoch: 12 | Batch Status: 45000/50000 (90%) | Loss: 1.147684
Train Epoch: 12 | Batch Status: 45500/50000 (91%) | Loss: 1.331857
Train Epoch: 12 | Batch Status: 46000/50000 (92%) | Loss: 1.146997
Train Epoch: 12 | Batch Status: 46500/50000 (93%) | Loss: 1.455274
Train Epoch: 12 | Batch Status: 47000/50000 (94%) | Loss: 0.899112
Train Epoch: 12 | Batch Status: 47500/50000 (95%) | Loss: 1.134018
Train Epoch: 12 | Batch Status: 48000/50000 (96%) | Loss: 1.405144
Train Epoch: 12 | Batch Status: 48500/50000 (97%) | Loss: 1.378047
Train Epoch: 12 | Batch Status: 49000/50000 (98%) | Loss: 1.091374
Train Epoch: 12 | Batch Status: 49500/50000 (99%) | Loss: 1.270836
===========================
Test set: Average loss: 0.0269, Accuracy: 5246/10000 (52%)
Train Epoch: 13 | Batch Status: 0/50000 (0%) | Loss: 1.040835
Train Epoch: 13 | Batch Status: 500/50000 (1%) | Loss: 1.150819
Train Epoch: 13 | Batch Status: 1000/50000 (2%) | Loss: 1.139843
Train Epoch: 13 | Batch Status: 1500/50000 (3%) | Loss: 1.064028
Train Epoch: 13 | Batch Status: 2000/50000 (4%) | Loss: 1.120252
Train Epoch: 13 | Batch Status: 2500/50000 (5%) | Loss: 1.105386
Train Epoch: 13 | Batch Status: 3000/50000 (6%) | Loss: 1.233262
Train Epoch: 13 | Batch Status: 3500/50000 (7%) | Loss: 1.276505
Train Epoch: 13 | Batch Status: 4000/50000 (8%) | Loss: 1.078591
Train Epoch: 13 | Batch Status: 4500/50000 (9%) | Loss: 1.083709
Train Epoch: 13 | Batch Status: 5000/50000 (10%) | Loss: 0.962368
Train Epoch: 13 | Batch Status: 5500/50000 (11%) | Loss: 1.302445
Train Epoch: 13 | Batch Status: 6000/50000 (12%) | Loss: 1.045652
Train Epoch: 13 | Batch Status: 6500/50000 (13%) | Loss: 1.221993
Train Epoch: 13 | Batch Status: 7000/50000 (14%) | Loss: 1.140236
Train Epoch: 13 | Batch Status: 7500/50000 (15%) | Loss: 1.433411
Train Epoch: 13 | Batch Status: 8000/50000 (16%) | Loss: 1.138477
Train Epoch: 13 | Batch Status: 8500/50000 (17%) | Loss: 1.148220
Train Epoch: 13 | Batch Status: 9000/50000 (18%) | Loss: 1.427309
Train Epoch: 13 | Batch Status: 9500/50000 (19%) | Loss: 1.320654
Train Epoch: 13 | Batch Status: 10000/50000 (20%) | Loss: 1.104850
Train Epoch: 13 | Batch Status: 10500/50000 (21%) | Loss: 0.945989
Train Epoch: 13 | Batch Status: 11000/50000 (22%) | Loss: 0.935194
Train Epoch: 13 | Batch Status: 11500/50000 (23%) | Loss: 1.390335
Train Epoch: 13 | Batch Status: 12000/50000 (24%) | Loss: 0.927915
Train Epoch: 13 | Batch Status: 12500/50000 (25%) | Loss: 1.127218
Train Epoch: 13 | Batch Status: 13000/50000 (26%) | Loss: 1.132138
Train Epoch: 13 | Batch Status: 13500/50000 (27%) | Loss: 1.090854
Train Epoch: 13 | Batch Status: 14000/50000 (28%) | Loss: 1.022005
Train Epoch: 13 | Batch Status: 14500/50000 (29%) | Loss: 1.346144
Train Epoch: 13 | Batch Status: 15000/50000 (30%) | Loss: 1.271944
Train Epoch: 13 | Batch Status: 15500/50000 (31%) | Loss: 1.386978
Train Epoch: 13 | Batch Status: 16000/50000 (32%) | Loss: 1.445673
Train Epoch: 13 | Batch Status: 16500/50000 (33%) | Loss: 0.988743
Train Epoch: 13 | Batch Status: 17000/50000 (34%) | Loss: 1.111568
Train Epoch: 13 | Batch Status: 17500/50000 (35%) | Loss: 1.333954
Train Epoch: 13 | Batch Status: 18000/50000 (36%) | Loss: 1.107365
Train Epoch: 13 | Batch Status: 18500/50000 (37%) | Loss: 1.001128
Train Epoch: 13 | Batch Status: 19000/50000 (38%) | Loss: 1.115523
Train Epoch: 13 | Batch Status: 19500/50000 (39%) | Loss: 1.282103
Train Epoch: 13 | Batch Status: 20000/50000 (40%) | Loss: 1.137964
Train Epoch: 13 | Batch Status: 20500/50000 (41%) | Loss: 1.340389
Train Epoch: 13 | Batch Status: 21000/50000 (42%) | Loss: 1.064469
Train Epoch: 13 | Batch Status: 21500/50000 (43%) | Loss: 1.061508
Train Epoch: 13 | Batch Status: 22000/50000 (44%) | Loss: 0.958167
Train Epoch: 13 | Batch Status: 22500/50000 (45%) | Loss: 1.369749
Train Epoch: 13 | Batch Status: 23000/50000 (46%) | Loss: 1.114118
Train Epoch: 13 | Batch Status: 23500/50000 (47%) | Loss: 1.356488
Train Epoch: 13 | Batch Status: 24000/50000 (48%) | Loss: 1.274715
Train Epoch: 13 | Batch Status: 24500/50000 (49%) | Loss: 1.179663
Train Epoch: 13 | Batch Status: 25000/50000 (50%) | Loss: 1.158439
Train Epoch: 13 | Batch Status: 25500/50000 (51%) | Loss: 1.175449
Train Epoch: 13 | Batch Status: 26000/50000 (52%) | Loss: 1.069482
Train Epoch: 13 | Batch Status: 26500/50000 (53%) | Loss: 1.058106
Train Epoch: 13 | Batch Status: 27000/50000 (54%) | Loss: 1.015354
Train Epoch: 13 | Batch Status: 27500/50000 (55%) | Loss: 1.203759
Train Epoch: 13 | Batch Status: 28000/50000 (56%) | Loss: 1.059842
Train Epoch: 13 | Batch Status: 28500/50000 (57%) | Loss: 1.303600
Train Epoch: 13 | Batch Status: 29000/50000 (58%) | Loss: 1.240651
Train Epoch: 13 | Batch Status: 29500/50000 (59%) | Loss: 1.270092
Train Epoch: 13 | Batch Status: 30000/50000 (60%) | Loss: 1.007402
Train Epoch: 13 | Batch Status: 30500/50000 (61%) | Loss: 1.099290
Train Epoch: 13 | Batch Status: 31000/50000 (62%) | Loss: 1.183896
Train Epoch: 13 | Batch Status: 31500/50000 (63%) | Loss: 1.589356
Train Epoch: 13 | Batch Status: 32000/50000 (64%) | Loss: 1.197910
Train Epoch: 13 | Batch Status: 32500/50000 (65%) | Loss: 1.047356
Train Epoch: 13 | Batch Status: 33000/50000 (66%) | Loss: 1.215995
Train Epoch: 13 | Batch Status: 33500/50000 (67%) | Loss: 1.210836
Train Epoch: 13 | Batch Status: 34000/50000 (68%) | Loss: 0.960552
Train Epoch: 13 | Batch Status: 34500/50000 (69%) | Loss: 0.967662
Train Epoch: 13 | Batch Status: 35000/50000 (70%) | Loss: 1.168923
Train Epoch: 13 | Batch Status: 35500/50000 (71%) | Loss: 1.201773
Train Epoch: 13 | Batch Status: 36000/50000 (72%) | Loss: 1.090146
Train Epoch: 13 | Batch Status: 36500/50000 (73%) | Loss: 1.207967
Train Epoch: 13 | Batch Status: 37000/50000 (74%) | Loss: 1.140089
Train Epoch: 13 | Batch Status: 37500/50000 (75%) | Loss: 1.189663
Train Epoch: 13 | Batch Status: 38000/50000 (76%) | Loss: 1.035644
Train Epoch: 13 | Batch Status: 38500/50000 (77%) | Loss: 1.103113
Train Epoch: 13 | Batch Status: 39000/50000 (78%) | Loss: 1.082593
Train Epoch: 13 | Batch Status: 39500/50000 (79%) | Loss: 1.269657
Train Epoch: 13 | Batch Status: 40000/50000 (80%) | Loss: 1.398881
Train Epoch: 13 | Batch Status: 40500/50000 (81%) | Loss: 1.481868
Train Epoch: 13 | Batch Status: 41000/50000 (82%) | Loss: 1.227347
Train Epoch: 13 | Batch Status: 41500/50000 (83%) | Loss: 1.125766
Train Epoch: 13 | Batch Status: 42000/50000 (84%) | Loss: 1.310667
Train Epoch: 13 | Batch Status: 42500/50000 (85%) | Loss: 1.318879
Train Epoch: 13 | Batch Status: 43000/50000 (86%) | Loss: 1.141818
Train Epoch: 13 | Batch Status: 43500/50000 (87%) | Loss: 1.017269
Train Epoch: 13 | Batch Status: 44000/50000 (88%) | Loss: 1.229960
Train Epoch: 13 | Batch Status: 44500/50000 (89%) | Loss: 1.123035
Train Epoch: 13 | Batch Status: 45000/50000 (90%) | Loss: 1.115651
Train Epoch: 13 | Batch Status: 45500/50000 (91%) | Loss: 0.958296
Train Epoch: 13 | Batch Status: 46000/50000 (92%) | Loss: 0.982767
Train Epoch: 13 | Batch Status: 46500/50000 (93%) | Loss: 1.310734
Train Epoch: 13 | Batch Status: 47000/50000 (94%) | Loss: 1.471834
Train Epoch: 13 | Batch Status: 47500/50000 (95%) | Loss: 1.085164
Train Epoch: 13 | Batch Status: 48000/50000 (96%) | Loss: 1.247231
Train Epoch: 13 | Batch Status: 48500/50000 (97%) | Loss: 1.197501
Train Epoch: 13 | Batch Status: 49000/50000 (98%) | Loss: 0.972283
Train Epoch: 13 | Batch Status: 49500/50000 (99%) | Loss: 1.310326
===========================
Test set: Average loss: 0.0272, Accuracy: 5214/10000 (52%)
Train Epoch: 14 | Batch Status: 0/50000 (0%) | Loss: 0.965832
Train Epoch: 14 | Batch Status: 500/50000 (1%) | Loss: 1.049312
Train Epoch: 14 | Batch Status: 1000/50000 (2%) | Loss: 1.095589
Train Epoch: 14 | Batch Status: 1500/50000 (3%) | Loss: 0.940827
Train Epoch: 14 | Batch Status: 2000/50000 (4%) | Loss: 0.966843
Train Epoch: 14 | Batch Status: 2500/50000 (5%) | Loss: 1.019123
Train Epoch: 14 | Batch Status: 3000/50000 (6%) | Loss: 1.244878
Train Epoch: 14 | Batch Status: 3500/50000 (7%) | Loss: 1.240441
Train Epoch: 14 | Batch Status: 4000/50000 (8%) | Loss: 1.085171
Train Epoch: 14 | Batch Status: 4500/50000 (9%) | Loss: 0.903064
Train Epoch: 14 | Batch Status: 5000/50000 (10%) | Loss: 1.152579
Train Epoch: 14 | Batch Status: 5500/50000 (11%) | Loss: 1.060572
Train Epoch: 14 | Batch Status: 6000/50000 (12%) | Loss: 0.968121
Train Epoch: 14 | Batch Status: 6500/50000 (13%) | Loss: 1.162034
Train Epoch: 14 | Batch Status: 7000/50000 (14%) | Loss: 1.230020
Train Epoch: 14 | Batch Status: 7500/50000 (15%) | Loss: 0.629958
Train Epoch: 14 | Batch Status: 8000/50000 (16%) | Loss: 1.100350
Train Epoch: 14 | Batch Status: 8500/50000 (17%) | Loss: 1.344887
Train Epoch: 14 | Batch Status: 9000/50000 (18%) | Loss: 1.214938
Train Epoch: 14 | Batch Status: 9500/50000 (19%) | Loss: 1.159198
Train Epoch: 14 | Batch Status: 10000/50000 (20%) | Loss: 0.942264
Train Epoch: 14 | Batch Status: 10500/50000 (21%) | Loss: 1.141669
Train Epoch: 14 | Batch Status: 11000/50000 (22%) | Loss: 1.247305
Train Epoch: 14 | Batch Status: 11500/50000 (23%) | Loss: 1.036190
Train Epoch: 14 | Batch Status: 12000/50000 (24%) | Loss: 0.998038
Train Epoch: 14 | Batch Status: 12500/50000 (25%) | Loss: 1.057495
Train Epoch: 14 | Batch Status: 13000/50000 (26%) | Loss: 1.126798
Train Epoch: 14 | Batch Status: 13500/50000 (27%) | Loss: 0.943086
Train Epoch: 14 | Batch Status: 14000/50000 (28%) | Loss: 1.001570
Train Epoch: 14 | Batch Status: 14500/50000 (29%) | Loss: 1.330795
Train Epoch: 14 | Batch Status: 15000/50000 (30%) | Loss: 1.211850
Train Epoch: 14 | Batch Status: 15500/50000 (31%) | Loss: 1.435869
Train Epoch: 14 | Batch Status: 16000/50000 (32%) | Loss: 0.988005
Train Epoch: 14 | Batch Status: 16500/50000 (33%) | Loss: 1.500458
Train Epoch: 14 | Batch Status: 17000/50000 (34%) | Loss: 1.245832
Train Epoch: 14 | Batch Status: 17500/50000 (35%) | Loss: 1.010616
Train Epoch: 14 | Batch Status: 18000/50000 (36%) | Loss: 0.993630
Train Epoch: 14 | Batch Status: 18500/50000 (37%) | Loss: 1.016443
Train Epoch: 14 | Batch Status: 19000/50000 (38%) | Loss: 1.095472
Train Epoch: 14 | Batch Status: 19500/50000 (39%) | Loss: 1.025755
Train Epoch: 14 | Batch Status: 20000/50000 (40%) | Loss: 1.208151
Train Epoch: 14 | Batch Status: 20500/50000 (41%) | Loss: 0.943733
Train Epoch: 14 | Batch Status: 21000/50000 (42%) | Loss: 1.180969
Train Epoch: 14 | Batch Status: 21500/50000 (43%) | Loss: 1.331183
Train Epoch: 14 | Batch Status: 22000/50000 (44%) | Loss: 1.080585
Train Epoch: 14 | Batch Status: 22500/50000 (45%) | Loss: 0.845648
Train Epoch: 14 | Batch Status: 23000/50000 (46%) | Loss: 0.900813
Train Epoch: 14 | Batch Status: 23500/50000 (47%) | Loss: 1.198422
Train Epoch: 14 | Batch Status: 24000/50000 (48%) | Loss: 1.115273
Train Epoch: 14 | Batch Status: 24500/50000 (49%) | Loss: 1.300904
Train Epoch: 14 | Batch Status: 25000/50000 (50%) | Loss: 1.052230
Train Epoch: 14 | Batch Status: 25500/50000 (51%) | Loss: 0.991227
Train Epoch: 14 | Batch Status: 26000/50000 (52%) | Loss: 1.130112
Train Epoch: 14 | Batch Status: 26500/50000 (53%) | Loss: 1.102052
Train Epoch: 14 | Batch Status: 27000/50000 (54%) | Loss: 1.154985
Train Epoch: 14 | Batch Status: 27500/50000 (55%) | Loss: 0.901053
Train Epoch: 14 | Batch Status: 28000/50000 (56%) | Loss: 1.386838
Train Epoch: 14 | Batch Status: 28500/50000 (57%) | Loss: 1.318704
Train Epoch: 14 | Batch Status: 29000/50000 (58%) | Loss: 1.165538
Train Epoch: 14 | Batch Status: 29500/50000 (59%) | Loss: 1.286547
Train Epoch: 14 | Batch Status: 30000/50000 (60%) | Loss: 1.019864
Train Epoch: 14 | Batch Status: 30500/50000 (61%) | Loss: 0.933961
Train Epoch: 14 | Batch Status: 31000/50000 (62%) | Loss: 1.371692
Train Epoch: 14 | Batch Status: 31500/50000 (63%) | Loss: 0.869138
Train Epoch: 14 | Batch Status: 32000/50000 (64%) | Loss: 1.343452
Train Epoch: 14 | Batch Status: 32500/50000 (65%) | Loss: 1.148451
Train Epoch: 14 | Batch Status: 33000/50000 (66%) | Loss: 1.461880
Train Epoch: 14 | Batch Status: 33500/50000 (67%) | Loss: 1.310805
Train Epoch: 14 | Batch Status: 34000/50000 (68%) | Loss: 1.037715
Train Epoch: 14 | Batch Status: 34500/50000 (69%) | Loss: 1.108068
Train Epoch: 14 | Batch Status: 35000/50000 (70%) | Loss: 1.221787
Train Epoch: 14 | Batch Status: 35500/50000 (71%) | Loss: 1.250794
Train Epoch: 14 | Batch Status: 36000/50000 (72%) | Loss: 1.104398
Train Epoch: 14 | Batch Status: 36500/50000 (73%) | Loss: 0.936922
Train Epoch: 14 | Batch Status: 37000/50000 (74%) | Loss: 1.254037
Train Epoch: 14 | Batch Status: 37500/50000 (75%) | Loss: 1.390203
Train Epoch: 14 | Batch Status: 38000/50000 (76%) | Loss: 0.922697
Train Epoch: 14 | Batch Status: 38500/50000 (77%) | Loss: 1.029495
Train Epoch: 14 | Batch Status: 39000/50000 (78%) | Loss: 1.276032
Train Epoch: 14 | Batch Status: 39500/50000 (79%) | Loss: 1.481597
Train Epoch: 14 | Batch Status: 40000/50000 (80%) | Loss: 1.117967
Train Epoch: 14 | Batch Status: 40500/50000 (81%) | Loss: 1.313189
Train Epoch: 14 | Batch Status: 41000/50000 (82%) | Loss: 0.854444
Train Epoch: 14 | Batch Status: 41500/50000 (83%) | Loss: 0.985908
Train Epoch: 14 | Batch Status: 42000/50000 (84%) | Loss: 1.222997
Train Epoch: 14 | Batch Status: 42500/50000 (85%) | Loss: 1.547109
Train Epoch: 14 | Batch Status: 43000/50000 (86%) | Loss: 1.144234
Train Epoch: 14 | Batch Status: 43500/50000 (87%) | Loss: 0.998672
Train Epoch: 14 | Batch Status: 44000/50000 (88%) | Loss: 1.359176
Train Epoch: 14 | Batch Status: 44500/50000 (89%) | Loss: 1.203832
Train Epoch: 14 | Batch Status: 45000/50000 (90%) | Loss: 1.072643
Train Epoch: 14 | Batch Status: 45500/50000 (91%) | Loss: 0.914735
Train Epoch: 14 | Batch Status: 46000/50000 (92%) | Loss: 0.920592
Train Epoch: 14 | Batch Status: 46500/50000 (93%) | Loss: 1.018140
Train Epoch: 14 | Batch Status: 47000/50000 (94%) | Loss: 1.018603
Train Epoch: 14 | Batch Status: 47500/50000 (95%) | Loss: 0.929566
Train Epoch: 14 | Batch Status: 48000/50000 (96%) | Loss: 1.208977
Train Epoch: 14 | Batch Status: 48500/50000 (97%) | Loss: 0.781861
Train Epoch: 14 | Batch Status: 49000/50000 (98%) | Loss: 1.013169
Train Epoch: 14 | Batch Status: 49500/50000 (99%) | Loss: 1.159188
===========================
Test set: Average loss: 0.0273, Accuracy: 5307/10000 (53%)
'''
