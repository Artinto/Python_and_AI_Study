from torch import nn
import torch
from torch import tensor

x_data = tensor([[1.0], [2.0], [3.0]])
y_data = tensor([[2.0], [4.0], [6.0]]) #x,y의 데이터를 3x1행렬로 입력


class Model(nn.Module): # 신경망 모듈. 매개변수를 캡슐화(encapsulation)하는 간편한 방법 으로, GPU로 이동, 내보내기(exporting), 불러오기(loading) 등의 작업을 위한 헬퍼(helper)를 제공합니다.
    def __init__(self): #생성자
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(Model, self).__init__() #부모클래스
        self.linear = torch.nn.Linear(1, 1)  # One in and one out 단순 선형 회귀 input_dim=1, output_dim=1.

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        y_pred = self.linear(x) #객체를 호출하면 자동으로 x를대입하여 forward 연산이 수행됩니다.
        return y_pred


# our model
model = Model()

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(reduction='sum') #loss 함수로는 MSE(평균제곱로차)사용, 출력값은 다 더함
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)    #optimizer 클래스 초기화
                                                            #SGD 경사하강법 # parameter() 변수 # 0.01 학습률 

# Training loop 
for epoch in range(500):                                    
    # 1) Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data) #x_data 값으로 forward 값 y_pred 에 입력
 
    # 2) Compute and print loss
    loss = criterion(y_pred, y_data) #평균제곱오차값 loss 입력
    print(f'Epoch: {epoch} | Loss: {loss.item()} ') #횟수 , loss값 출력

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()   #경사하강법 변수 초기화, 모든 변화도를 0으로
    loss.backward()         #loss 역전파
    optimizer.step()        #step()이란 함수를 실행시키면 우리가 미리 선언할 때 
                            # 지정해 준 model의 파라미터들이 업데이트 된다.


# After training
hour_var = tensor([[4.0]])
y_pred = model(hour_var)
print("Prediction (after training)",  4, model(hour_var).data[0][0].item())# After training
