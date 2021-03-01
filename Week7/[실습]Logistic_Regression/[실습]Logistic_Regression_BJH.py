from torch import nn
import numpy as np
import torch
from torch import tensor
from torch import sigmoid
import torch.optim as optim
import torch.nn.functional as F

csv = np.loadtxt('diabetes.csv', delimiter=",", dtype=np.float32)
x_data = tensor(csv[:,0:8])
y_data = tensor(csv[:,8:9])

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()  
        self.linear = torch.nn.Linear(8, 1)  

    def forward(self, x):
        y_pred = sigmoid(self.linear(x)) 
        return y_pred
i=0
count=0
model = Model()
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

for epoch in range(1000):
    
    y_pred = model(x_data) 
    loss = criterion(y_pred, y_data)
    print(f'Epoch: {epoch} | Loss: {loss.item()} ') 
    optimizer.zero_grad() 
    loss.backward() 
    optimizer.step()

for i in range(len(y_data)):
    if y_pred[i] > 0.5:
        answer=1
    else:
        answer=0
    if answer==y_data[i]:
        count+=1
 
hour_var = x_data[0] #이걸 tensor(x_data[0])으로 돌릴시 user에러가 뜸
y_pred = model(hour_var)
print("Prediction (after training)",  x_data[0] ,":", y_pred.item())
print("accuracy : ",(count/len(x_data))*100)
  
       
