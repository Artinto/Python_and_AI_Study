from torch import nn, optim, from_numpy
import numpy as np

xy = np.loadtxt('diabetes.csv', delimiter=',', dtype=np.float32)
x_data = from_numpy(xy[:, 0:-1]) # ndarray type -> Tensor type
y_data = from_numpy(xy[:, [-1]]) # ndarray type -> Tensor type
print(f'X\'s shape: {x_data.shape} | Y\'s shape: {y_data.shape}') # .format() 함수와 비슷하게 직접 중괄호 안에 넣어도 된다.


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = nn.Linear(8, 6) # 1 번째 Layer
        self.l2 = nn.Linear(6, 4) # 2 번째 Layer
        self.l3 = nn.Linear(4, 1) # 3 번째 Layer

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out1 = self.sigmoid(self.l1(x))  # 1번째 Layer에 대한 Actiavtion func 출력 
        out2 = self.sigmoid(self.l2(out1)) # 2번째 Layer에 대한 Actiavtion func 출력 
        y_pred = self.sigmoid(self.l3(out2)) # 3번째 Layer에 대한 Actiavtion func 출력 -> y_prediction 
        return y_pred


model = Model()

criterion = nn.BCELoss(reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(100):
    y_pred = model(x_data)

    loss = criterion(y_pred, y_data)
    print(f'Epoch: {epoch + 1}/100 | Loss: {loss.item():.4f}') # loss.item():.4f -> 소수 넷째 자리 까지 출력 

    optimizer.zero_grad() # Gradient 초기화
    loss.backward()  # Backpropagation을 이용하여 Gradient 구하기 
    optimizer.step() # Weight Update
