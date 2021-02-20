from torch import tensor #배열(numpy와 비슷하지만 더 빠름)
from torch import nn #신경망 학습 패키지
from torch import sigmoid #로지스틱 회귀의 loss함수인 시그모이드
import torch.nn.functional as F #nn모듈의 함수
import torch.optim as optim #최적화 알고리즘 패키지

# Training data and ground truth
x_data = tensor([[1.0], [2.0], [3.0], [4.0]])
y_data = tensor([[0.], [0.], [1.], [1.]]) #바이너리 데이터(0,1)


class Model(nn.Module): #nn.Module의 서브클래스로 새 모듈을 정의(상속)
    def __init__(self):
        """
        In the constructor we instantiate nn.Linear module
        """
        super(Model, self).__init__() #부모 클래스인 nn.Module의 생성자 호출
        self.linear = nn.Linear(1, 1)  #선형모듈 생성 1개의 인풋 1개의 아웃풋

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data.
        """
        y_pred = sigmoid(self.linear(x)) #선언된 선형모듈을 시그모이드함수의 변수로 입력(예측값)
        return y_pred


# our model
model = Model() #생성자 호출

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = nn.BCELoss(reduction='mean') #binary_cross_entropy
optimizer = optim.SGD(model.parameters(), lr=0.01) #sgd(확률적 경사하강법)의 생성자에 model.parameters()를 호출하면 모델의
                                                    멤버인 nn.Linear 모듈의 학습가능한 매개변수들이 포함됨

# Training loop
for epoch in range(1000):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data) #예측값

    # Compute and print loss
    loss = criterion(y_pred, y_data) #손실계산
    print(f'Epoch {epoch + 1}/1000 | Loss: {loss.item():.4f}')

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad() #변화도를 0으로 만들고, 역전파 수행후 가중치 업데이트
    loss.backward()
    optimizer.step()

# After training
print(f'\nLet\'s predict the hours need to score above 50%\n{"=" * 50}')
hour_var = model(tensor([[1.0]]))
print(f'Prediction after 1 hour of training: {hour_var.item():.4f} | Above 50%: {hour_var.item() > 0.5}')
hour_var = model(tensor([[7.0]]))
print(f'Prediction after 7 hours of training: {hour_var.item():.4f} | Above 50%: { hour_var.item() > 0.5}')
