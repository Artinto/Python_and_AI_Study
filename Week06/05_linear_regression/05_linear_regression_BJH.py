from torch import nn
import torch
from torch import tensor

x_data = tensor([[1.0], [2.0], [3.0]])
y_data = tensor([[2.0], [4.0], [6.0]]) #y=2x인 일차원 


class Model(nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(Model, self).__init__() #상속 
        self.linear = torch.nn.Linear(1, 1)  # y=2x인 일차원 이기에 하나의 x 하나의 y 

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        y_pred = self.linear(x)
        return y_pred


# our model
model = Model()

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(reduction='sum') #MSELOSS의 출력값을 전부 더하여 criterion에 저장 
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) #경사하강법을 이용하여 가중치 갱신 

# Training loop
for epoch in range(500):
    # 1) Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data) # y^ 
    
    # 2) Compute and print loss 
    loss = criterion(y_pred, y_data) #위에서 sum으로 선언하였기에 더하여 loss에 저장
    print(f'Epoch: {epoch} | Loss: {loss.item()} ') # loss출력 

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad() # 기울기 초기화
    loss.backward() #역전파 
    optimizer.step()


# After training
hour_var = tensor([[4.0]])
y_pred = model(hour_var)
print("Prediction (after training)",  4, model(hour_var))
