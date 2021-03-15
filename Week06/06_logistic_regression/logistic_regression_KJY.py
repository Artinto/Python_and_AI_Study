from torch import tensor
from torch import nn
from torch import sigmoid # sigmoid 
import torch.nn.functional as F # 각종 함수들이 들어있는 모듈 -> Non-linear activation functions 중 하나인 sigmoid 를 사용하기 위함
import torch.optim as optim # 최적화 함수 생성할 때 사용할 모듈 이름 간소화

# Training data and ground truth
x_data = tensor([[1.0], [2.0], [3.0], [4.0]])
y_data = tensor([[0.], [0.], [1.], [1.]]) # logistic regression 가 2진수를 이용하여 이루어지기 때문에 y_data 는 0,1

class Model(nn.Module): # Model 클래스는 torch.nn.Module 의 서브 클래스이다
    def __init__(self): # constructor
        super(Model, self).__init__() # 부모 클래스 torch.nn.Module 의 __init__ 매소드 호출
        # nn.Linear()는 입력의 차원, 출력의 차원을 인수로 받습니다.
        # 받은 아이를 self.linear 에 저장~
        self.linear=nn.Linear(1,1) # input value x:1, output value y:1
        
    def forward(self, x):
        y_pred = sigmoid(self.linear(x)) # 1. constructor 에서 생성한 self.linear함수에 input 값 집어넣어서 output 값 도출
                                           # 2. sigmoid 함수를 사용하여 wrapping
        return y_pred # 위에서 구한 값을 return

# our model
model = Model()

criterion = nn.BCELoss(reduction='mean') # Binary Cross Entropy Loss 
# reduction='' : 작은 따옴표 안에 들어가는 내용이 어떤 형태로 변형시킬 것인지 정해줌
# ex) none : 변형 없음
# ex) mean : output 의 평균값 냄
# ex) sum : output 의 합계 냄
# default 값은 mean이다.
# cross entropy 를 통해 손실 함수 구현

optimizer=optim.SGD(model.parameters(), lr=0.01)
#실제로 많이 사용되는 가장 단순한 가중치 갱신 규칙은 확률적 경사하강법(SGD; Stochastic Gradient Descent)
#model.parameters()-> 갱신할 변수를 공급
#learning rate



#Training loop
for epoch in range(500):
    # Forward pass : Compute predicted y by passing x to the model
    y_pred=model(x_data)
    # print(x_data, y_pred) -> input 과 output 확인가능
    # compute and print loss
   
    loss=criterion(y_pred, y_data) # 손실함수 통해 오차 구함
    print(f'Epoch {epoch + 1}/1000 | Loss: {loss.item():.4f}') # 몇 번째 training 인가 + 손실함수에 의한 오차 출력, 오차는 소수점밑4자리까지 출력
    
    #optimizer.zero_grad() 를 사용하여 수동으로 변화도 버퍼를 0으로 설정하는 것에 유의해야함 
    # 이는 역전파(Backprop) 과정에서 변화도가 누적되기 때문
    optimizer.zero_grad()
    loss.backward() # 역전파(Backprop)
    optimizer.step() # optimizer 에 의해 parameter를 갱신 update 함수는 optimizer.step()
    
# After training 
print(f'\nLet\'s predict the hours need to score above 50%\n{"=" * 50}') 
hour_var = model(tensor([[1.0]])) # 학습 후 1.0 을 넣으면
print(f'Prediction after 1 hour of training: {hour_var.item():.4f} | Above 50%: {hour_var.item() > 0.5}') # 예상되는 결과는 ?
hour_var = model(tensor([[7.0]])) # 학습 후 7.0 을 넣으면
print(f'Prediction after 7 hours of training: {hour_var.item():.4f} | Above 50%: { hour_var.item() > 0.5}') # 예상되는 결과는?
