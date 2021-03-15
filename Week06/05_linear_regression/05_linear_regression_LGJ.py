from torch import nn  #torch에서 nn을 불러옴(Neural Network)
import torch  #pytorch를 불러옴
from torch import tensor  #torch 에서 tensor를 불러옴

x_data = tensor([[1.0], [2.0], [3.0]])  #3x1의 2차원 텐서(벡터)선언
y_data = tensor([[2.0], [4.0], [6.0]])  #3x1의 2차원 텐서(벡터)선언


class Model(nn.Module): #nn.Module의 하위 클래스인 Model이라는 클래스를 생성 
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(Model, self).__init__() #nn.Module의 생성자를 호출
        self.linear = torch.nn.Linear(1, 1)  #One in and one out #객체에 입력,출력이 각각 1개인 선형 모듈을 적용

    def forward(self, x): #x라는 한개의 변수를 받는 foward라는 클래스 생성
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        y_pred = self.linear(x) #y_pred에 self.linear(x)의 값을 대입
        return y_pred #y.pred 반환


# our model
model = Model() #model에 Model클래스를 적용

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(reduction='sum') #criterion에 평균제곱오차값 대입
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  #optimizer에 확률적경사하강법(Stochastic Gradient Decent)을 사용, parameters()함수를 통해 변수 자동입력, 학습률은 0.01

# Training loop
for epoch in range(500):  #경사하강법 500번 실행
    # 1) Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data)  #model에 x_data를 대입하여 나온 foward의 결괏값을 y_data에 대입

    # 2) Compute and print loss
    loss = criterion(y_pred, y_data)  #loss에 y_pred와 y_data의 평균제곱오차값을 대입
    print(f'Epoch: {epoch} | Loss: {loss.item()} ') #SGD횟수,loss값을 출력

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad() #경사하강법 재실행을 위해 미분했던 값 초기화
    loss.backward() #loss에 대해 역전파 시행(w,b에 대한 loss의 편미분 구하기)
    optimizer.step()  


# After training
hour_var = tensor([[4.0]])  #학습 후 x=4를 대입했을 때의 결괏값을 
y_pred = model(hour_var)
print("Prediction (after training)",  4, model(hour_var).data[0][0].item())# After training
