from torch import nn, optim, from_numpy 
import numpy as np

xy = np.loadtxt('./data/diabetes.csv.gz', delimiter=',', dtype=np.float32)  # xy의 데이터를 넘파이형식으로 로드함. 경로'./data/diabetes.csv.gz', 구분자 ',' 
x_data = from_numpy(xy[:, 0:-1])  # xy 데이터셋의 맨 마지막 열을 제하고 나머지를 x_data로 지정
y_data = from_numpy(xy[:, [-1]])  # xy 데이터셋의 맨 마지막 열만 y_data로 지정
print(f'X\'s shape: {x_data.shape} | Y\'s shape: {y_data.shape}') #


class Model(nn.Module): # nn.Module의 상속을 받는 Model 클래스를 생성
    def __init__(self): #초기화 메서드
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(Model, self).__init__() # nn.Module의 생성자를 호출
        self.l1 = nn.Linear(8, 6) # input 8, output 6
        self.l2 = nn.Linear(6, 4) # input 6, output 4
        self.l3 = nn.Linear(4, 1) # input 4, output 1

        self.sigmoid = nn.Sigmoid() # self.sigmoid에 신경망의 시그모이드함수 할당

    def forward(self, x): #x라는 한개의 변수를 받는 foward라는 클래스 생성
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        out1 = self.sigmoid(self.l1(x)) # x에 관한 선형함수, 시그모이드 함수를 차례로 거친 결과
        out2 = self.sigmoid(self.l2(out1))  # out1에 관한 선형함수, 시그모이드 함수를 차례로 거친 결과
        y_pred = self.sigmoid(self.l3(out2))  # out2에 관한 선형함수, 시그모이드 함수를 차례로 거친 결과
        return y_pred 


# our model
model = Model() 


# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = nn.BCELoss(reduction='mean')  # criterion에 BCE(Binary Cross Entropy,분류기)오차를 대입
optimizer = optim.SGD(model.parameters(), lr=0.1) # 최적화 함수로 경사하강법을 사용

# Training loop
for epoch in range(100):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data)

    # Compute and print loss
    loss = criterion(y_pred, y_data)
    print(f'Epoch: {epoch + 1}/100 | Loss: {loss.item():.4f}')

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
