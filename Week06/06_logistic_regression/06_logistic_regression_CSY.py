# torch에서 데이터의 배열인 tensor를 불러옴
# torch 모듈 내에서 neural network의 약자인 nn을 불러옴
# torch 모듈 내에서 sigmoid 함수를 불러옴
#  torch.nn은 클래스로 정의됨, torch.nn.functional은 함수로 정의됨 
# optim 패키지를 불러옴
from torch import tensor
from torch import nn
from torch import sigmoid
import torch.nn.functional as F
import torch.optim as optim

# Training data and ground truth
# x_data와 y_data에 각각 3 X 1 데이터 형식의 2차원 tensor로 정의함/logistic regression이기 때문에 y_data는 0과 1로 이루어짐
x_data = tensor([[1.0], [2.0], [3.0], [4.0]])
y_data = tensor([[0.], [0.], [1.], [1.]])

# torch.nn.Module을 상속받는 파이썬 클래스 정의
# 모델의 생성자 정의 (객체가 갖는 속성값을 초기화하는 역할,객체가 생성될 때 자동으로 호출됨)
class Model(nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate nn.Linear module
        """
        # super() 함수를 통해 nn.Module의 생성자를 호출함
        # nn.Linear() 함수를 이용하여 비선형 모델을 만듦(1차원의 입력과 1차원의 출력을 인수로 받음->하나의 입력 데이터인 x값에 대해서 하나의 출력 데이터인 y값이 나오기 때문)
        super(Model, self).__init__()
        self.linear = nn.Linear(1, 1)  # One in and one out

    # 모델 객체와 학습 데이터인 x를 받아 forward 연산하는 함수로 model(입력 데이터) 형식으로 객체를 호출하면 자동으로 forward 연산 수행됨
    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data.
        """
        # x를 인자로 넘겨 받아 비선형 모델을 통해 계산된 값을 시그모이드 활성화 함수를 통해 얻은 값을 y_pred에 저장
        # y_pred 값 반환
        y_pred = sigmoid(self.linear(x))
        return y_pred


# our model
# Model 클래스 변수 model 선언
model = Model()

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
# 손실함수 정의 코드로, 손실 함수로 BCE(Binary Cross Entropy)를 사용함. reduction은 감소 또는 가공 데이터의 각 batch(데이터 샘플들의 묶음)의 값을 집계하는 방법으로 output의 평균 계산
# SGD(확률적 경사 하강법)는 경사 하강법의 일종이고 model.parameters()를 이용하여 W와 b를 전달함. lr은 학습률이며 이는 경사하강법을 설정하는 코드 
criterion = nn.BCELoss(reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
# 학습을 1000번 반복
for epoch in range(1000):
    # Forward pass: Compute predicted y by passing x to the model
    # x_data를 모델에 인자로 전달하여 y_pred 값을 계산함
    y_pred = model(x_data)

    # Compute and print loss
    # y_pred와 y_data를 인자로 넘겨 받아 손실함수를 계산하여 저장
    # 학습을 반복한 횟수와 손실함수 값(소수 넷째자리까지)을 출력
    loss = criterion(y_pred, y_data)
    print(f'Epoch {epoch + 1}/1000 | Loss: {loss.item():.4f}')

    # Zero gradients, perform a backward pass, and update the weights.
    # 미분을 통해 얻은 기울기가 이전에 계산된 기울기 값에 누적되기 때문에 기울기 값을 0으로 초기화하여 새로운 (가중치와 편향)에 대한 새로운 기울기를 구할 수 있도록 함
    # 역전파 실행, 손실 함수를 미분하여 기울기 계산
    # step() 함수를 호출하여 W와 b를 업데이트함
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# After training
# 학습을 끝낸 후 주어진 테스트 값인 시간에 따라서 50% 이상의 점수를 맞으면 True 아니면 False가 출력됨
# 입력 값 1.0에 대한 예측값을 받아  hour_var에 저장
# 입력 값 7.0에 대한 예측값을 받아  hour_var에 저장
print(f'\nLet\'s predict the hours need to score above 50%\n{"=" * 50}')
hour_var = model(tensor([[1.0]]))
print(f'Prediction after 1 hour of training: {hour_var.item():.4f} | Above 50%: {hour_var.item() > 0.5}')
hour_var = model(tensor([[7.0]]))
print(f'Prediction after 7 hours of training: {hour_var.item():.4f} | Above 50%: { hour_var.item() > 0.5}')
