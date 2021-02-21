# class를 사용해 모델 만들기
# optimizer.step()를 사용해서 업데이트
# 이외의 이전코드와 다른 점 거의 없음.

from torch import nn
import torch
from torch import tensor

# tensor형 Matrix를 사용한 데이터 선언
x_data = tensor([[1.0], [2.0], [3.0]])
y_data = tensor([[2.0], [4.0], [6.0]])
# tensor 행렬계산등에 더 적합한 형태

# ----------------------------------------------------------------------------

# class로 모델만들때 nn.Module상속과 super.__init__은 꼭 해야하는 부분

# class로 model 만들기
class Model(nn.Module): # nn.Module을 상속받아 사용함.
    # torch.nn.Module : input받은 tensor를 처리하여 output해주는 containeer / nn모듈을 가져다가 사용하고 싶을때 사용
    # torch.nn : 딥러닝에 필요한 모듈이 모아져있는 패키지 (loss func, layer(model), ...)
    def __init__(self): # 초기설정
        
        super(Model, self).__init__() 
        # = torch.nn.Module.__init__()

        self.linear = torch.nn.Linear(1, 1)  # torch.nn의 Linear모듈을 가져다 사용하는 모습
        # nn.Linear()에는 w와 b가 저장되어 있으며 이는 parameters()로 불러올 수 있다.
        # Linear(input size, output size)

    def forward(self, x): # input x size : 1
        """
        학습데이터를 입력받아서 forward propagation을 진행시키는 함수.
        반드시 forward라는 이름이어야 함.
        """
        y_pred = self.linear(x) # 선언했던 Linear 모듈 사용

        #__init__에서 선언했던 torch.nn 모듈(loss function, layer...)들을 조합하여 모델 디자인
        
        return y_pred  # output y_pred size : 1

# -----------------------------------------------------------------------------

# 학습에 필요한 model, loss, optimizer 선언
model = Model()

criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# model.parameters() : model(Linear)의 파라미터(w,b)를 반환해줌.


# -----------------------------------------------------------------------------

# 학습시작
for epoch in range(500): # 500번 반복학습
    # 1) Forward pass
    y_pred = model(x_data)

    # 2) Compute and print loss
    loss = criterion(y_pred, y_data) 
    print(f'Epoch: {epoch} | Loss: {loss.item()} ')

    # 매 epoch마다 gradient가 이어서 연산되는 걸 막기 위해 zero_grad()로 초기화
    optimizer.zero_grad()
    loss.backward()    # backward(backpropagation) : computational graph 내의 모든 변수가 loss에 미치는 영향(gradient)을 계산
    optimizer.step()   # update
    # model.parameters()를 통해 제공된 파라미터(w, b)를 한번에 update. (matrix연산)

# 학습 후, test
# Q. 4시간 공부했을 때(x = 4), 몇 점을 맞겠는가(y = ?)
hour_var = tensor([[4.0]])
y_pred = model(hour_var)
print("Prediction (after training)",  4, model(hour_var).data[0][0].item())
