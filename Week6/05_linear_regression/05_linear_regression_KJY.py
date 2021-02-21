import torch
from torch import nn
from torch import tensor

x_data=tensor([[1.0], [2.0], [3.0]])
y_data=tensor([[2.0], [4.0], [6.0]])

class Model(nn.Module): # Model 클래스는 torch.nn.Module 의 서브 클래스이다
    def __init__(self): # constructor
        super(Model, self).__init__() # 부모 클래스 torch.nn.Module 의 __init__ 매소드 호출
        
        # nn.Linear()는 입력의 차원, 출력의 차원을 인수로 받습니다.
        # 받은 아이를 self.linear 에 저장~
        self.linear=torch.nn.Linear(1,1) # input value x:1, output value y:1
    def forward(self,x):
        y_pred=self.linear(x) # constructor 에서 생성한 self.linear함수에 input 값 집어넣어서 output 값 도출
        return y_pred
    
model=Model() # instance 생성
 
criterion=torch.nn.MSELoss(reduction='sum') # reduction='' : 작은 따옴표 안에 들어가는 내용이 어떤 형태로 변형시킬 것인지 정해줌
# ex) none : 변형 없음
# ex) mean : output 의 평균값 냄
# ex) sum : output 의 합계 냄
# default 값은 mean이다.
#출력과 대상간의 평균제곱오차(mean-squared error)를 계산하는 손실함수(nn 모듈 내장)

optimizer=torch.optim.SGD(model.parameters(), lr=0.01)
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
    print(f'epoch: {epoch} | Loss: {loss.item()}') # 몇 번째 training 인가 + 손실함수에 의한 오차 출력
    
    #optimizer.zero_grad() 를 사용하여 수동으로 변화도 버퍼를 0으로 설정하는 것에 유의해야함
    # 이는 역전파(Backprop) 과정에서 변화도가 누적되기 때문
    optimizer.zero_grad()
    loss.backward() # 역전파(Backprop)
    optimizer.step() # optimizer 에 의해 parameter를 갱신 update 함수는 optimizer.step()
    
    #test 코드
    hour_var=tensor([[4.0]])
    y_pred=model(hour_var)
    print("predict (after training)", 4, model(hour_var).data[0][0].item()) # 4넣었을 때 훈련된 모델으로부터 예상되는 값은?
