from torch import nn, optim, from_numpy
import numpy as np

xy = np.loadtxt('./data/diabetes.csv', delimiter=',', dtype=np.float32) #32비트를 통한 실수 표현
x_data = from_numpy(xy[:, 0:-1]) #-1은 끝자리를 의미
y_data = from_numpy(xy[:, [-1]]) #맨끝의 데이터만 추출하여 y_data로 사용
print(f'X\'s shape: {x_data.shape} | Y\'s shape: {y_data.shape}')


class Model(nn.Module): #Model의 기능 상속
    def __init__(self): #초기 설정
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(Model, self).__init__()
        self.l1 = nn.Linear(8, 6) # 8개의 인풋 6개의 아웃풋 첫번째 뉴런
        self.l2 = nn.Linear(6, 4) # 6개의 인풋 4개의 아웃풋 두번째 뉴런
        self.l3 = nn.Linear(4, 1) # 4개의 인풋 1개의 아웃풋 세번째 뉴런

        self.sigmoid = nn.Sigmoid() # 시그노이드 계산

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        out1 = self.sigmoid(self.l1(x)) #첫번째 뉴런의 계산 결과
        out2 = self.sigmoid(self.l2(out1)) #두번째 뉴런의 계산 결과
        y_pred = self.sigmoid(self.l3(out2)) # out2와 그의 가중치를 곱한 것이 y_pred 마지막 
        return y_pred


# our model
model = Model()


# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = nn.BCELoss(reduction='mean') #다 더한 후 갯수만큼 나눠주기
optimizer = optim.SGD(model.parameters(), lr=0.1) #경사하단법

# Training loop
for epoch in range(100):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data)

    # Compute and print loss
    loss = criterion(y_pred, y_data)
    print(f'Epoch: {epoch + 1}/100 | Loss: {loss.item():.4f}')

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()# 미분을 통해 얻은 기울기 초기화
    loss.backward()  #역전파 시행
    optimizer.step() #가중치 업데이트
