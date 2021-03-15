from torch import tensor#torch에서 tensor모듈 가져옴
from torch import nn#nn가져옴
from torch import sigmoid#sigmoid 가져옴
import torch.nn.functional as F#nn.functional을 f라고 칭함
import torch.optim as optim#torch.optim을 optim이라 칭함

# Training data and ground truth
x_data = tensor([[1.0], [2.0], [3.0], [4.0]])#x , y좌표를 각각 1차원 행렬 즉 2차원 tensor로 정의
y_data = tensor([[0.], [0.], [1.], [1.]])


class Model(nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate nn.Linear module
        """
        super(Model, self).__init__()#super을 통해 nn.Module을 호출하여 초기화 함
        self.linear = nn.Linear(1, 1)  # One in and one out 선행함수를 사용하여 1차원 입력과 1차원 출력을 할 수 있는 모델을 만듬

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data.
        """
        y_pred = sigmoid(self.linear(x))#계산된 값을 sigmoid활성화 함수를 통해 계산된 값을  저장
        return y_pred


# our model
model = Model()#클래스 선언 

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = nn.BCELoss(reduction='mean')#손실 함수 bce를 사용하고 reduction은 가공 데이터의 값을 집계하는 방식으로 평균을 계산 
optimizer = optim.SGD(model.parameters(), lr=0.01)#확률적 경사 하강법인 sgd를 사용하여 파라미터를 이용하여 w와 b의 값을 받고 학습률이 0.01이다 

# Training loop
for epoch in range(1000):#학습을 총 1000번
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data)#예측값 선언

    # Compute and print loss
    loss = criterion(y_pred, y_data)#손실함수를 계산한다
    print(f'Epoch {epoch + 1}/1000 | Loss: {loss.item():.4f}')

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()#변화율은 계속 누적이 되기 때문에 변화율을 0으로 만든다
    loss.backward()#역전파를 통해 기울기 계산
    optimizer.step()#wb를 업데이트 

# After training
print(f'\nLet\'s predict the hours need to score above 50%\n{"=" * 50}')#테스트 값이 시간에 따라서 50%이상이면 true 아니면 false
hour_var = model(tensor([[1.0]]))#입력값 1
print(f'Prediction after 1 hour of training: {hour_var.item():.4f} | Above 50%: {hour_var.item() > 0.5}')
hour_var = model(tensor([[7.0]]))#입력값 7
print(f'Prediction after 7 hours of training: {hour_var.item():.4f} | Above 50%: { hour_var.item() > 0.
