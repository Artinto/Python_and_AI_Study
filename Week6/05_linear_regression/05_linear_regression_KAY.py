from torch import nn#torch안의 nn 모듈 사용
import torch
from torch import tensor#torch 안의 tensor 사용

x_data = tensor([[1.0], [2.0], [3.0]])#데이터를 행렬에 저
y_data = tensor([[2.0], [4.0], [6.0]])


class Model(nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(Model, self).__init__()#nn.Module.__init__()을 실행시키며 초기화 함, super를 통해 nn.Module을 가져옴
        self.linear = torch.nn.Linear(1, 1)  # One in and one out # x한개당 y하나 linear을 통해 선형 모델을 만듦

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        y_pred = self.linear(x)#선형 모델을 통해 예측값 저
        return y_pred


# our model
model = Model()#Model클라스 사용

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(reduction='sum')#손실함수 mseloss는 대상간의 평균제곱 오차를 계산,reduction은 데이터의 묶음을 뜻하며 다 더함
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)#확률적 경사 하강법 파라미터를 통해서 w와 b의 값을 주며  학습률이 0,01이다

# Training loop
for epoch in range(500):#학습 500번
    # 1) Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data)#xdata를 입력하여 y예측값 출력

    # 2) Compute and print loss
    loss = criterion(y_pred, y_data)#손실함수 계산하여 저장
    print(f'Epoch: {epoch} | Loss: {loss.item()} ')#f_string {}안의 값은 변수이

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()#기울기의 값이 누적되기 때문에 기울기 값을 0으로 만들어준다
    loss.backward()#역전파를 실행하여 손실 함수를 미분하여 계산
    optimizer.step()#step함수를 통해 w와 b의 값을 바


# After training
hour_var = tensor([[4.0]])#학습후 임의의 값 4를 입력
y_pred = model(hour_var)#예측 값을 저
print("Prediction (after training)",  4, model(hour_var).data[0][0].item())# After training#예측값 출력
