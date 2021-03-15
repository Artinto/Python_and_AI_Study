from torch import nn, optim, from_numpy #torch의 nn, optim, from_numpy 가져오기
import numpy as np

xy = np.loadtxt('/content/drive/MyDrive/diabetes.csv', delimiter=',', dtype=np.float32) #numpy 데이터 불러오기
x_data = from_numpy(xy[:, 0:-1]) # [전체 열, 마지막(-1)전까지] 슬라이싱
y_data = from_numpy(xy[:, [-1]]) # [전체 데이터, 마지막하나만 가져옴]
'''
# (numpy array를) tensor형으로 변환하기
torch.Tensor() : Numpy array의 사본. 따라서 tensor의 값을 변경하여도 Numpy array의 값이 달라지지 않음. 
torch.from_numpy() : input array의 dtype을 그대로 받고 tensor와 메모리를 공유함. tensor의 값이 변경되면 Numpy array값도 변경됨.
'''

print(f'X\'s shape: {x_data.shape} | Y\'s shape: {y_data.shape}')
# 출력 : X's shape: torch.Size([759, 8]) | Y's shape: torch.Size([759, 1])

'''
# f-string : python3.6이상 지원하는 문자열 포맷팅으로
    > print(f'hello {name}')로 사용할 수 있고,
    기존의 str.format과는 다르게 정수의 상술연산도 지원함.
    > print(f'sum: {a+b}')
'''


class Model(nn.Module): # nn.Module을 통해 모델만들기
    def __init__(self):
       
        super(Model, self).__init__()
        self.l1 = nn.Linear(8, 6) # input : 8
        self.l2 = nn.Linear(6, 4)
        self.l3 = nn.Linear(4, 1) # output : 1
        # linear모델을 3개를 쌓아서 보다 deep한 모델을 만듦.

        # 나머지 가운데의 사이즈는 이전layer와 이후 layer의 input, output사이즈만 맞춰주면 됨. 자유

        self.sigmoid = nn.Sigmoid() 
        # activation함수로 sigmoid사용.(torch.nn 모듈에 class로 짜여있음.)
        # BCELoss에는 0~1의 수가 들어가야 하므로 sigmoid를 거친 후에 넣어줘야함.

    def forward(self, x):
        out1 = self.sigmoid(self.l1(x))
        out2 = self.sigmoid(self.l2(out1))
        y_pred = self.sigmoid(self.l3(out2))
        # deep
        return y_pred


# our model
model = Model()

criterion = nn.BCELoss(reduction='mean')
# 마지막 layer의 노드 수가 1되는 binary classification의 경우, BCE로스 + activation F을 사용함.
# cf) CrossEntropyLoss는 softmax를 포함하고 있으며 마지막 layer의 노드 수가 2 이상이어햐 한다.

optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training loop
for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(f'Epoch: {epoch + 1}/100 | Loss: {loss.item():.4f}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
