# pandas 를 이용한 데이터 추출
import pandas as pd
import numpy as np
from torch import nn
import torch
from torch import tensor

csv_test=pd.read_csv('C:/input_file/data-01-test-score/data-01-test-score.csv', header=None, names=['A','B','C','D'])
x=csv_test[['A','B','C']]

y=csv_test['D']
x_data = torch.FloatTensor(x.values)

y_data = torch.FloatTensor(y.values)

y_data = y_data.unsqueeze(-1)



class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear=torch.nn.Linear(3,1)
    
    def forward(self, x):
        y_pred=self.linear(x)
        return y_pred
model=Model()

criterion=torch.nn.MSELoss(reduction='mean')
optimizer=torch.optim.SGD(model.parameters(), lr=0.00001)

for epoch in range(1000):
    y_pred=model(x_data)
    loss=criterion(y_pred,y_data)
    print(f'Epoch: {epoch} | Loss: {loss.item()} ')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

hour_var=tensor([73.0, 80.0, 75.0])
y_pred=model(hour_var)
print("Prediction (after training)", 73.0, 80.0, 75.0, ":", y_pred.item())
