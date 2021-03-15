# 기존 linear모델에 Sigmoid함수를 추가하여 binary한 문제의 logistic regression 모델을 만듦.

from torch import tensor
from torch import nn
from torch import sigmoid # 함수
#import torch.nn.functional as F
import torch.optim as optim


# tensor로 데이터 선언
x_data = tensor([[1.0], [2.0], [3.0], [4.0]])
y_data = tensor([[0.], [0.], [1.], [1.]])
# binary(0 or 1)한 y데이터

# ----------------------------------------------------------------------------

# class로 모델만들때 nn.Module상속과 super.__init__은 꼭 해야하는 부분

# class로 model 만들기
class Model(nn.Module): # nn.Module을 상속받아 사용함.
    # torch.nn.Module : input받은 tensor를 처리하여 output해주는 containeer / nn모듈을 가져다가 사용하고 싶을때 사용
    # torch.nn : 딥러닝에 필요한 모듈이 모아져있는 패키지 (loss func, layer(model), ...)
    def __init__(self): # 초기설정
        
        super(Model, self).__init__() 
        # = torch.nn.Module.__init__()

        self.linear = nn.Linear(1, 1)  # torch.nn의 Linear모듈을 가져다 사용하는 모습
        # nn.Linear()에는 w와 b가 저장되어 있으며 이는 parameters()로 불러올 수 있다.
        # Linear(input size, output size)

    def forward(self, x): # input x size : 1
        """
        학습데이터를 입력받아서 forward propagation을 진행시키는 함수.
        반드시 forward라는 이름이어야 함.
        """
        y_pred = sigmoid(self.linear(x)) # x > linear > sigmoid 통과
        # sigmoid()는 class가 아니라 torch에 있는 함수임(따라서 위에서 선언 안함)

        #__init__에서 선언했던 torch.nn 모듈(loss function, layer...)들을 조합하여 모델 디자인
        
        return y_pred  # output y_pred size : 1

# -----------------------------------------------------------------------------

# 학습에 필요한 model, loss, optimizer 선언
model = Model()

criterion = nn.BCELoss(reduction='mean') # binary cross entropy loss
optimizer = optim.SGD(model.parameters(), lr=0.01)
# model.parameters() : model(Linear)의 파라미터(w,b)를 반환해줌.


# -----------------------------------------------------------------------------

# 학습시작
for epoch in range(1000): # 1000번 반복학습
    
    y_pred = model(x_data) # Forward 
    loss = criterion(y_pred, y_data) 
    print(f'Epoch {epoch + 1}/1000 | Loss: {loss.item():.4f}') # 소수점 넷째자리까지 표시(.4f)

    # 매 epoch마다 gradient가 이어서 연산되는 걸 막기 위해 zero_grad()로 초기화
    optimizer.zero_grad()
    loss.backward()    # Backward(backpropagation) : computational graph 내의 모든 변수가 loss에 미치는 영향(gradient)을 계산
    optimizer.step()   # update
    # model.parameters()를 통해 제공된 파라미터(w, b)를 한번에 update. 

# 학습 후, test
# Q. 50점 이상 받으려면 몇 시간 공부해야 할까?
print(f'\nLet\'s predict the hours need to score above 50%\n{"=" * 50}')
hour_var = model(tensor([[1.0]]))  # 1시간 공부했을 때(x=1)
print(f'Prediction after 1 hour of training: {hour_var.item():.4f} | Above 50%: {hour_var.item() > 0.5}') # False (hour_var.item() > 0.5 을 만족하지 않음)
hour_var = model(tensor([[7.0]]))  # 7시간 공부했을 때(x=7)
print(f'Prediction after 7 hours of training: {hour_var.item():.4f} | Above 50%: { hour_var.item() > 0.5}') # True (hour_var.item() > 0.5 을 만족함)
