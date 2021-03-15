from torch import tensor  #torch에서 tensor를 사용
from torch import nn  #torch에서 신경망(Neural network) 사용
from torch import sigmoid #torch에서 sigmoid함수 사용
import torch.nn.functional as F #Logistic Regression의 Cost(Loss)함수를 사용
import torch.optim as optim #torch에서 optim 사용

# Training data and ground truth
x_data = tensor([[1.0], [2.0], [3.0], [4.0]]) #x_data: 4x1의 2차원 행렬
y_data = tensor([[0.], [0.], [1.], [1.]]) #y_data: 4x1의 2차원 행렬


class Model(nn.Module): #Module의 하위 클래스인 Model 생성
    def __init__(self):
        """
        In the constructor we instantiate nn.Linear module
        """
        super(Model, self).__init__() #nn.Module의 생성자 호출
        self.linear = nn.Linear(1, 1)  # One in and one out #객체에 입력,출력이 각각 1개인 선형 모듈을 적용

    def forward(self, x): #x라는 한개의 변수를 받는 foward라는 클래스 생성
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data.
        """
        y_pred = sigmoid(self.linear(x))  #y_pred에 sigmoid(self.linear(x))의 값을 대입
        return y_pred #y.pred 반환


# our model
model = Model() #model에 Model클래스를 적용

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = nn.BCELoss(reduction='mean')  #criterion에 BCE(Binary Cross Entropy,분류기)오차를 대입
optimizer = optim.SGD(model.parameters(), lr=0.01)  #optimizer에 확률적경사하강법 대입

# Training loop
for epoch in range(1000): #경사하강법을 1000번 실행
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data)  #model에 x_data를 대입하여 나온 foward의 결괏값을 y_data에 대입

    # Compute and print loss
    loss = criterion(y_pred, y_data)  #loss에 y_pred와 y_data의 분류기(BCE)값을 대입
    print(f'Epoch {epoch + 1}/1000 | Loss: {loss.item():.4f}')  

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad() #optimizer의 자동미분 값을 0으로 만들어줌
    loss.backward() #loss 역전파
    optimizer.step()  #경사 하강법 후의 w값과 b값을 갱신

# After training
print(f'\nLet\'s predict the hours need to score above 50%\n{"=" * 50}')  #50%이상의 점수를 맞기위해 필요한 시간을 예측합시다=======
hour_var = model(tensor([[1.0]]))
print(f'Prediction after 1 hour of training: {hour_var.item():.4f} | Above 50%: {hour_var.item() > 0.5}')
hour_var = model(tensor([[7.0]]))
print(f'Prediction after 7 hours of training: {hour_var.item():.4f} | Above 50%: { hour_var.item() > 0.5}')
