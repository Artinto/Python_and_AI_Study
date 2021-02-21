from torch import nn     # 신경망을 생성할 수 있는 패키지
import torch
from torch import tensor # tensor : 계산 속도를 빠르게 하기 위해 GPU를 사용할 수 있게 만들어줌

x_data = tensor([[1.0], [2.0], [3.0]]) # 입력 tensor list
y_data = tensor([[2.0], [4.0], [6.0]]) # 결과 tensor list


class Model(nn.Module): # nn.Module을 상속받는  Model class 생성
    def __init__(self): # initial
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(Model, self).__init__()        # 부모 클래스 생성자 호출
        self.linear = torch.nn.Linear(1, 1)  # One in and one out
        # nn.Linear : 파이토치에 구현되어 있는 선형 회귀 모델
        # parameter : first > 입력 차원 second > 출력 차원

    def forward(self, x): # 선형 모델 계산
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        y_pred = self.linear(x)
        return y_pred


# our model
model = Model() # Model 객체 생성

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.

# loss 함수로 MSE(Mean Squared Error)를 사용
# reduction='sum' : 출력값은 모두 더한다
criterion = torch.nn.MSELoss(reduction='sum')

# optimizer setting
# 경사 하강법인 SGD를 사용하고 learning rate는 2.01로 설정
# model.parameters()는 w(weight)와 b(bias)를 전달
optimizer = torch.optim.SGD(model.parameters(), lr=2.01)

# Training loop
for epoch in range(500): # == for(i = 0; i < 500; i++)
    # 1) Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data) # 예측값을 저장

    # 2) Compute and print loss
    # 예측값(y_pred)와 결과값(y_data)의 loss를 구함
    loss = criterion(y_pred, y_data)
    print(f'Epoch: {epoch} | Loss: {loss.item()} ')

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad() # backword 하기 전 gradient를 초기화 (cuz, gradient가 덮여쓰여지지 않고 누적되기 때문)
    loss.backward()       # loss에 대한 backword 진행
    optimizer.step()      # parameter 초기화


# After training
hour_var = tensor([[4.0]]) # 학습 후 4.0을 입력 해 결과값 도출
y_pred = model(hour_var)
print("Prediction (after training)",  4, model(hour_var).data[0][0].item())# After training