from torch import tensor
from torch import nn          #신경망
from torch import sigmoid      #로지스틱 회귀 시그모이드
import torch.nn.functional as F #nn모듈의 함수
import torch.optim as optim

# Training data and ground truth
x_data = tensor([[1.0], [2.0], [3.0], [4.0]])
y_data = tensor([[0.], [0.], [1.], [1.]])


class Model(nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate nn.Linear module
        """
        super(Model, self).__init__() #부모클래스
        self.linear = nn.Linear(1, 1)  # One in and one out 단순 선형 회귀

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data.
        """
        y_pred = sigmoid(self.linear(x)) # self.linear(X)를 sigmoid함수에 입력
        return y_pred


# our model
model = Model()

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = nn.BCELoss(reduction='mean')        #손실함수를 BCELoss 함수를 사용   
                                                #binary 다중 분류 손실 함수
                                                #reduction='mean' 평균값.
optimizer = optim.SGD(model.parameters(), lr=0.01) #경사하강법

# Training loop
for epoch in range(1000):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data) #x_data를 Model에 입력하여 나온 출력값을 입력

    # Compute and print loss
    loss = criterion(y_pred, y_data) #BCE함수로 구한 로스를 입력
    print(f'Epoch {epoch + 1}/1000 | Loss: {loss.item():.4f}') #횟수/1000  ,로스 소수점 4자리까지

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad() #경사하강법 변수 초기화, 모든 변화도를 0으로
    loss.backward()         #loss 역전파
    optimizer.step()         #step()이란 함수를 실행시키면 우리가 미리 선언할 때 
                            # 지정해 준 model의 파라미터들이 업데이트 된다.  

# After training
print(f'\nLet\'s predict the hours need to score above 50%\n{"=" * 50}')
hour_var = model(tensor([[1.0]]))
print(f'Prediction after 1 hour of training: {hour_var.item():.4f} | Above 50%: {hour_var.item() > 0.5}')
hour_var = model(tensor([[7.0]]))
print(f'Prediction after 7 hours of training: {hour_var.item():.4f} | Above 50%: { hour_var.item() > 0.5}')
#결과 값이 0.5 이상이면 True를 출력하고 0.5 미만이면 False를 출력합니다.
