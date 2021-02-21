from torch import nn #torch 안에 nn이라는 모듈을 사용   torch 모듈사용 torch안에 tensor이라는 모듈사용
import torch 
from torch import tensor

x_data = tensor([[1.0], [2.0], [3.0]])   # x,y의 데이터를 3x1행렬로 각각 정의 그리고 벡터형식으로 나타냄
y_data = tensor([[2.0], [4.0], [6.0]])


class Model(nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(Model, self).__init__()#self를 안 써도됨 이것도 생성자 초기화 방법

        self.linear = torch.nn.Linear(1, 1)  # One in and one out  1인 1실 x 1개에 y 1개값이 나오게 선형함수를 설정한다

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        y_pred = self.linear(x) # y = wx +b의 식을 임의로 만들어냄 
        
        return y_pred


# our model
model = Model()  #Model 클래스 사용

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(reduction='sum') #mse 함수를 가져와 계산을 해줌
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)#학습률이 0.01인 확률적 경사하강법에 매개변수를넣어 학습시킴

# Training loop
for epoch in range(500): #500번 반복
    # 1) Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data)  #forward 값을 만듬

    # 2) Compute and print loss
    loss = criterion(y_pred, y_data) # loss를 만들어냄
    print(f'Epoch: {epoch} | Loss: {loss.item()} ') 

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()#변화율을 0으로 만들고 초기화 후 업데이트한다.
    optimizer.step()


# After training
hour_var = tensor([[+4.0]])  #1x1 행렬
y_pred = model(hour_var)  #반복실행후 나온 forward값 
print("Prediction (after training)",  4, model(hour_var).data[0][0].item())
