import numpy as np
from torch import nn, optim, from_numpy 
# from_numpy : numpy.ndarray(N차원의 배열 객체로, 같은 type 의 데이터로 구성) 로부터 Tensor 생성하도록 해주는 모듈
# optim : 최적화 알고리즘이 구현된 함수를 가진 모듈 (.zero_grad, .step)
# nn : neural network 에 필요한 각종 함수, class 를 가진 모듈

xy=np.loadtxt('C:/input_file/diabetes/diabetes.csv',delimiter=',',dtype=np.float32) # diabetes 파일 경로를 이용해 불러옴
# delimiter : 구분 문자는 ',' -> csv문서는 ','로 데이터를 구분한다
# dtype : 배열 xy에 저장될 데이터의 타입 지정. 여기서는 np 모듈의 float32 타입 사용

x_data=from_numpy(xy[:,0:-1]) # 배열 x_data에 가장 왼쪽 열부터 오른쪽에서 두번째 열까지의 데이터만 저장, 2차원 배열
y_data=from_numpy(xy[:,[-1]]) # 배열 y_data에 가장 오른쪽 열의 데이터만 저장, 2차원 배열
print(f'X\'s shape: {x_data.shape} | Y\'s shape: {y_data.shape}') # x_data는 NxM , y_data는 Nx1 배열 
# shape -> 배열이 몇 행 몇 열 인지 알려줌

class Model(nn.Module): # nn.Module : 모든 neural network 모듈들을 위한 기반 
                        # 추후 사용할 attribute과 method 가짐 (.parameters() 나 .zero_grad() 갖고 있음)
#class 의 subclass 인 Model class 생성

    def __init__(self):
        super(Model, self).__init__() # 부모 클래스의 생성자 얻음
    
    # nn.Linear : 리팩토링('결과의 변경 없이 코드의 구조를 재조정함')
    # w와 b 값을 초기화 하고 y=wx+b 연산을 해줌
    # nn.Linear(8,1) 로 끝낼 수 있는 부분을 3번으로 나누어 deep 하게 구현
        self.l1=nn.Linear(8,6) # input: 8 output:6
        self.l2=nn.Linear(6,4) # input: 6 output:4
        self.l3=nn.Linear(4,1) # input: 4 output:1
    
        self.sigmoid=nn.Sigmoid() # sigmoid 함수를 만듦
    
    def forward(self, x): # forward propagation 3분할
        out1=self.sigmoid(self.l1(x))
        out2=self.sigmoid(self.l2(out1))
        y_pred=self.sigmoid(self.l3(out2))
        return y_pred

model=Model() # Model class의 인스턴스화

criterion=nn.BCELoss(reduction='mean') # BCELoss 클래스의 인스턴스 생성, Binary Cross Entropy 에 의해 구한 오차값의 평균을 구함
optimizer=optim.SGD(model.parameters(), lr=0.1) # SGD(stochastic gradient descent) 클래스의 인스턴스 생성 
# update 과정에서 SGD 알고리즘을 채택

for epoch in range(100):
    y_pred=model(x_data)
    loss=criterion(y_pred,y_data)
    print(f'Epoch: {epoch + 1}/100 | Loss: {loss.item():.4f}')

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
