from torch import nn, optim, from_numpy
import numpy as np

xy = np.loadtxt("./diabetes.csv", delimiter=',', dtype=np.float32)
x_data = from_numpy(xy[:, 0:-1])#  csv파일에서 x 데이터를와  y데이터를 추출 (y는 마지막값 x는 마지막값제외 모든값)
y_data = from_numpy(xy[:, [-1]])
print(f'X\'s shape: {x_data.shape} | Y\'s shape: {y_data.shape}')


class Model(nn.Module):
    def __init__(self):#초기화
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(Model, self).__init__()
        self.l1 = nn.Linear(8, 6)  #8 input 6 output
        self.l2 = nn.Linear(6, 4)  #6 input 4 output
        self.l3 = nn.Linear(4, 1)  #4 input 1 output

        self.sigmoid = nn.Sigmoid() #시그모이드 함수

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        out1 = self.sigmoid(self.l1(x)) #1. x에대해 선형함수를 시그모이드함수에 대입
        out2 = self.sigmoid(self.l2(out1))# 2. out1에대해서
        y_pred = self.sigmoid(self.l3(out2))#3. out2에대해서
        return y_pred


# our model
model = Model()


# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = nn.BCELoss(reduction='mean') #분류기를 이용해 평균값을 구함
optimizer = optim.SGD(model.parameters(), lr=0.1) #학습률이 0.1인 경사하강법

# Training loop
for epoch in range(100):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data)

    # Compute and print loss
    loss = criterion(y_pred, y_data)
    print(f'Epoch: {epoch + 1}/100 | Loss: {loss.item():.4f}')

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad() #0초기화 로스계산 b w 업데이트 후 다시반복
    loss.backward()
    optimizer.step()
