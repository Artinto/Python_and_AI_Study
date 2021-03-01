import numpy as np
from torch import tensor        # tensor : 계산 속도를 빠르게 하기 위해 GPU를 사용할 수 있게 만들어줌
from torch import nn            # 신경망을 생성할 수 있는 패키지
from torch import sigmoid       # 로지스틱 함수인 sigmoid 사용 
                                # linear model의 결과값을 sigmoid의 입력으로 넣어주면 0-1 사이의 값을 얻음
import torch.nn.functional as F # nn == class / functional == function(객체 생성필요 X)
import torch.optim as optim     # 최적화 알고리즘 정의

dataset_path="/diabetes.csv"
dataset=np.loadtxt(dataset_path, delimiter=',',dtype=np.float32)
x_data=tensor(dataset[:,0:8])
y_data=tensor(dataset[:,8:9])


class Model(nn.Module): # nn.Module을 상속받는  Model class 생성
    def __init__(self): # initial
        super(Model, self).__init__()  # 부모 클래스 생성자 호출
        self.linear = nn.Linear(8, 1)  # One in and one out

    def forward(self, x): # 선형 모델 계산
        y_pred = sigmoid(self.linear(x)) # sigmoid 의 결과값
        return y_pred


model = Model() # Model 객체 생성
criterion = nn.BCELoss(reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=0.01)

cnt = 0
for epoch in range(1001): # == for(i = 0; i < 1001; i++)
    y_pred = model(x_data) # 예측값 저장
    loss = criterion(y_pred, y_data)
    print(f'Epoch {epoch + 1}/1000 | Loss: {loss.item():.4f}')
    if(epoch % 100 == 0):
        for idx in range(len(y_data)):
            if(round(y_pred[idx].item()) == y_data[idx].item()): cnt+=1
        print("Accuracy : ", cnt / len(y_data) * 100)
        cnt = 0
    optimizer.zero_grad() # backword 하기 전 gradient를 초기화 (cuz, gradient가 덮여쓰여지지 않고 누적되기 때문)
    loss.backward()       # loss에 대한 backword 진행
    optimizer.step()      # parameter 초기화