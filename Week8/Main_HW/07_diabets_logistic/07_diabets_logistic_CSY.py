# torch 모듈 내에서 nn(신경망을 구축을 위한 데이터 구조, 레이어 등 정의되어 있음)과 optim 패키지를 불러옴.
# torch.from_numpy는 자동으로 input array의 dtype을 상속받고 tensor와 메모리 버퍼를 공유함. tensor 값이 변경되면 Numpy array값 변경됨.
# numpy 라이브러리를 불러옴
from torch import nn, optim, from_numpy
import numpy as np

# xy에 데이터를 numpy텍스트 파일로 로드하고 데이터는 콤마로 구분하게 하며 데이터 타입을 float 32로 설정
# x_data는 데이터에서 0번째부터 끝에서 2번째까지의 데이터를, y_data는 마지막 데이터를 말하며 tensor로 변환할 때 원래 메모리를 상속받음
# x_data, y_data의 shape 출력
xy = np.loadtxt('./data/diabetes.csv.gz', delimiter=',', dtype=np.float32)
x_data = from_numpy(xy[:, 0:-1])
y_data = from_numpy(xy[:, [-1]])
print(f'X\'s shape: {x_data.shape} | Y\'s shape: {y_data.shape}')

# torch.nn.Module을 상속받는 파이썬 클래스 정의
# 모델의 생성자 정의 (객체가 갖는 속성값을 초기화하는 역할, 객체가 생성될 때 자동으로 호출됨)
class Model(nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear module
        """
        # super() 함수를 통해 nn.Module의 생성자를 호출함
        # nn.Linear() 함수를 이용하여 3개 층의 모델을 만듦((입력,출력)8과 1은 고정)
        super(Model, self).__init__()
        self.l1 = nn.Linear(8, 6)
        self.l2 = nn.Linear(6, 4)
        self.l3 = nn.Linear(4, 1)

        # 신경망의 시그모이드 함수를 self.sigmoid에 저장
        self.sigmoid = nn.Sigmoid()

    # 모델 객체와 학습 데이터인 x를 받아 forward 연산하는 함수로 model(입력 데이터) 형식으로 객체를 호출하면 자동으로 forward 연산 수행됨
    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        # 각 층에 대해 sigmoid함수를 거친 결과값 저장
        out1 = self.sigmoid(self.l1(x))
        out2 = self.sigmoid(self.l2(out1))
        y_pred = self.sigmoid(self.l3(out2))
        return y_pred


# our model
# Model 클래스 변수 model 선언
model = Model()


# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
# 손실함수 정의 코드로, 손실 함수로 BCE(Binary Cross Entropy)를 사용함. reduction은 가공 데이터의 각 batch(데이터 샘플들의 묶음)의 값을 집계하는 방법으로 output의 평균 계산
# SGD(확률적 경사 하강법)는 경사 하강법의 일종이고 model.parameters()를 이용하여 parameter를 전달함. lr은 학습률이며 이는 경사하강법을 설정하는 코드
criterion = nn.BCELoss(reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training loop
# 학습을 100번 반복
for epoch in range(100):
    # Forward pass: Compute predicted y by passing x to the model
    # x_data를 모델에 인자로 전달하여 y_pred 값을 계산함
    y_pred = model(x_data)

    # Compute and print loss
    # y_pred와 y_data를 인자로 넘겨 받아 손실함수를 계산하여 저장
    # 학습을 반복한 횟수와 손실함수 값을 출력
    loss = criterion(y_pred, y_data)
    print(f'Epoch: {epoch + 1}/100 | Loss: {loss.item():.4f}')

    # Zero gradients, perform a backward pass, and update the weights.
    # 미분을 통해 얻은 기울기가 이전에 계산된 기울기 값에 누적되기 때문에 기울기 값을 0으로 초기화 함
    # 역전파 실행, 손실 함수를 미분하여 기울기 계산
    # step() 함수를 호출하여 parameter를 업데이트함
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
