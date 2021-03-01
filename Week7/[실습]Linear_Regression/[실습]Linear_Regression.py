# -*- coding: utf-8 -*-
"""Untitled2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1nbGkgNWOW_AjZeu1zaDd2LeQY7V_RjVj
"""

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
from torch import nn
import torch
from torch import tensor


score=pd.read_csv('/content/drive/MyDrive/data-01-test-score.csv',names=('A','B','C','D'))

X=score[['A','B','C']]
Y=score[['D']]
x_data=torch.FloatTensor(X.values)
y_data=torch.FloatTensor(Y.values) # .values 를 해야 값만을 가져올수있음

class Model(nn.Module): 
    def __init__(self): 
       
        super(Model, self).__init__() 
        self.linear = torch.nn.Linear(3, 1)  

    def forward(self, x):
        y_pred = self.linear(x) 
        return y_pred


model = Model()
criterion = torch.nn.MSELoss(reduction='sum') 
optimizer = torch.optim.SGD(model.parameters(), lr=0.0000001) # 너무 클시 제대로 동작하지않음   

for epoch in range(1000):                                    
  
    y_pred = model(x_data) 
 
    
    loss = criterion(y_pred, y_data)
    print(f'Epoch: {epoch} | Loss: {loss.item()} ') 

  
    optimizer.zero_grad() 
    loss.backward()        
    optimizer.step()        

hour_var = tensor([[73.0, 80.0, 75.0]])
y_pred = model(hour_var)
print("Prediction (after training)",  73.0, 80.0, 75.0, model(hour_var).data[0][0].item())