from torch import nn, optim, from_numpy
# torch.from_numpy() : tensor와 numpy list가 메모리 공유(call by reference)
# torch.Tensor()     : tensor는 numly list의 사본일 뿐  (call by value)
import numpy as np

xy = np.loadtxt('./data/diabetes.csv.gz', delimiter=',', dtype=np.float32)
x_data = from_numpy(xy[:, 0:-1]) # 0열 ~ 마지막 -1 열까지 저장
y_data = from_numpy(xy[:, [-1]]) # 마지막 열 저장
print(f'X\'s shape: {x_data.shape} | Y\'s shape: {y_data.shape}')


class Model(nn.Module): # nn.Module 상속받는 Model class
    def __init__(self): # constructor
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(Model, self).__init__() # initialize
        self.l1 = nn.Linear(8, 6)     # 입력 : 8차원 > 출력 : 6차원
        self.l2 = nn.Linear(6, 4)     # 입력 : 6차원 > 출력 : 4차원
        self.l3 = nn.Linear(4, 1)     # 입력 : 4차원 > 출력 : 1차원

        self.sigmoid = nn.Sigmoid()   # sigmoid

    def forward(self, x): # forward function
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        out1 = self.sigmoid(self.l1(x))
        out2 = self.sigmoid(self.l2(out1))   # out1의 결과를 out2의 input으로
        y_pred = self.sigmoid(self.l3(out2)) # out2의 결과를 out3의 input으로
        return y_pred


# our model
model = Model() # 인스턴스 할당


# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = nn.BCELoss(reduction='mean')          # Loss 함수로 BCELoss사용 (mean : 결과값들의 평균으로 값 저장)
optimizer = optim.SGD(model.parameters(), lr=0.1) #최적화 함수로 SGD사용 (learning rate = 0.1)

# Training loop
for epoch in range(100): # == for(int i = 0; i < 100; i++)
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data) # y^ 구하기

    # Compute and print loss
    loss = criterion(y_pred, y_data) # loss 계산
    print(f'Epoch: {epoch + 1}/100 | Loss: {loss.item():.4f}')

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()