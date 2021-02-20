from torch import nn
import torch
from torch import tensor

x_data = tensor([[1.0], [2.0], [3.0]])
y_data = tensor([[2.0], [4.0], [6.0]])


class Model(nn.Module): #nn.Module의 서브클래스로 새 모듈을 정의(상속)
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(Model, self).__init__() #부모 클래스인 nn.Module의 생성자 호출
        self.linear = torch.nn.Linear(1, 1) #선형모듈 생성 1개의 인풋 1개의 아웃풋

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        y_pred = self.linear(x) #예측값
        return y_pred


# our model
model = Model() #클래스 변수 생성

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(reduction='sum') #mse계산
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) #sgd(확률적 경사하강법)의 생성자에 model.parameters()를 호출하면 모델의
                                                          멤버인 nn.Linear 모듈의 학습가능한 매개변수들이 포함됨, lr=learning rate

# Training loop
for epoch in range(500):
    # 1) Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data) #자동으로 forward의 값이 저장됨

    # 2) Compute and print loss
    loss = criterion(y_pred, y_data) #손실계산
    print(f'Epoch: {epoch} | Loss: {loss.item()} ')

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad() #변화도를 0으로 만들고, 역전파 수행후 가중치 업데이트
    loss.backward()
    optimizer.step()


# After training
hour_var = tensor([[4.0]]) #4를 넣었을 때의 예측값
y_pred = model(hour_var)
print("Prediction (after training)",  4, model(hour_var).data[0][0].item())# After training
