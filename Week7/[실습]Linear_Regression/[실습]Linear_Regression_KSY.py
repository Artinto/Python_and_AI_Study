import pandas as pd
import numpy as np
from torch import nn
import torch
from torch import tensor
x=np.loadtxt("./data-01-test-score.csv",usecols=(0,1,2),delimiter=',',dtype=np.float32)
y=np.loadtxt("./data-01-test-score.csv",usecols=(3),delimiter=',',dtype=np.float32)

x_data = torch.FloatTensor(x)
y_data = torch.FloatTensor(y)
print(y_data)
y_data = y_data.unsqueeze(-1)
print(y_data)

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.linear=torch.nn.Linear(3,1)
    def forward(self,x):
        y_pred = self.linear(x)
        return y_pred
model=Model()
criterion=torch.nn.MSELoss(reduction='mean')
optimizer=torch.optim.SGD(model.parameters(),lr=0.000005)
for epoch in range(1000):
    y_pred = model(x_data)
    loss=criterion(y_pred,y_data)
    print(f'Epoch : {epoch}  Loss  : {loss.item()}')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
hour_var =tensor([73.0,80.0,75.0])
y_pred=model(hour_var)
print("Prediction (after training)", 73.0, 80.0, 75.0, ":",y_pred.item())
