from torch import nn, optim, from_numpy#torch에서 nn, optim, from_numpy를 가져옴
import numpy as np#numpy를 np로 표시

xy = np.loadtxt('./data/diabetes.csv.gz', delimiter=',', dtype=np.float32)#데이터를 가져와서 xy에 저장
x_data = from_numpy(xy[:, 0:-1])#제일 마지막 행을 제외하고 x_data에 저장한다. from_numpy는 원래 저장되어 있던 데이터를 tensor로 변환하는 것이다  
y_data = from_numpy(xy[:, [-1]])#제일 마지막 행을 y_data에 저장한다 from_numpy는 원래 저장되어 있던 데이터를 tensor로 변환하는 것이다  
print(f'X\'s shape: {x_data.shape} | Y\'s shape: {y_data.shape}')


class Model(nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(Model, self).__init__()#초기화
        self.l1 = nn.Linear(8, 6)#x데이터는 8개, y데이터는 6개
        self.l2 = nn.Linear(6, 4)#x데이터는 6개, y데이터는 4개
        self.l3 = nn.Linear(4, 1)#x데이터는 4개, y데이터는 1개

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        out1 = self.sigmoid(self.l1(x))#l1데이터로 sigmod함수를 사용
        out2 = self.sigmoid(self.l2(out1))#out1데이터로 sigmod함수 사용
        y_pred = self.sigmoid(self.l3(out2))#out2데이터로 sigmod함수 사용
        return y_pred


# our model
model = Model()


# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = nn.BCELoss(reduction='mean')#손실함수을 이용하고 평균은 구함
optimizer = optim.SGD(model.parameters(), lr=0.1)#학습률이 0.1인 확률적 경사하강법 사용

# Training loop
for epoch in range(100):#100번 학습
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data)#예측값 구함

    # Compute and print loss
    loss = criterion(y_pred, y_data)#loss구함
    print(f'Epoch: {epoch + 1}/100 | Loss: {loss.item():.4f}')

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()#변화율은 누적되기 때문에 변화율을 0으로 만든다
    loss.backward()#기울기 계산
    optimizer.step()#w, b를 업데이트
