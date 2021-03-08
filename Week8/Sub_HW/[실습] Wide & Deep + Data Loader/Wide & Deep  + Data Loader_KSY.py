from torch import nn, optim, from_numpy
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch.nn as nn
import torch.optim as optim
from torch import sigmoid
from torch import tensor
from torch.utils.data import TensorDataset, DataLoader
#모듈 써본것들은 다 넣어둠



xy = np.loadtxt("./diabetes.csv", delimiter=',', dtype=np.float32)
x_data = from_numpy(xy[:, 0:-1])#  csv파일에서 x 데이터를와  y데이터를 추출 (y는 마지막값 x는 마지막값제외 모든값)
y_data = from_numpy(xy[:, [-1]])
x_train, x_test, y_train, y_test = train_test_split(x_data,y_data,shuffle=False)
class DiabetesDataset(Dataset):
    def __init__(self):
        
        self.len = x_train.shape[0]
        self.x_train = x_train
        self.y_train = y_train
    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index]
    def __len__(self):
        return self.len

dataset = TensorDataset(x_train,y_train)
train_loader = DataLoader(dataset, batch_size=60, shuffle=True, num_workers=0)    
class Model(nn.Module):
    def __init__(self):#초기화
       
        super(Model, self).__init__()
        self.l1 = nn.Linear(8, 6)  
        self.l2 = nn.Linear(6, 4)  
        self.l3 = nn.Linear(4, 1)  

        self.sigmoid = nn.Sigmoid() #시그모이드 함수

    def forward(self, x):
        
        out1 = self.sigmoid(self.l1(x)) 
        out2 = self.sigmoid(self.l2(out1))
        y_pred = self.sigmoid(self.l3(out2))
        return y_pred

 
check=0
count=0
model = Model()
criterion = nn.BCELoss(reduction='mean') 
optimizer = optim.SGD(model.parameters(), lr=0.1) 

# Training loop
for epoch in range(1001):
    for i, data in enumerate(train_loader, 0): #인덱스= epoch
        check+=1
        inputs, labels = data
        inputs, labels = tensor(inputs), tensor(labels) #  텐서
        y_pred = model(inputs)
        loss=criterion(y_pred,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        if check % 1000 ==0:
            prediction = y_pred >= torch.FloatTensor([0.5])
            correct_prediction = prediction.float()==labels
            accuracy = correct_prediction.sum().item()/len(correct_prediction)
            print('Epoch {:4d}/{} loss: {:.6f} Accuracy {:2.2f}%'.format(epoch,1000,loss.item(),accuracy*100,))
            check=0
            
#로스값은 바뀌는데 accuracy 값이 동일함(?)         
#https://blog.naver.com/mjw4260/222225598018     
#batch 사이즈(?)



#오류 :  <ipython-input-2-6e750e49c936>:62: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
  inputs, labels = tensor(inputs), tensor(labels) #  텐서
  
  
Epoch   99/1000 loss: 0.666476 Accuracy 62.07%
Epoch  199/1000 loss: 0.569635 Accuracy 75.86%
Epoch  299/1000 loss: 0.629379 Accuracy 65.52%
Epoch  399/1000 loss: 0.408501 Accuracy 86.21%
Epoch  499/1000 loss: 0.628062 Accuracy 62.07%
Epoch  599/1000 loss: 0.443016 Accuracy 82.76%
Epoch  699/1000 loss: 0.352382 Accuracy 82.76%
Epoch  799/1000 loss: 0.409102 Accuracy 79.31%
Epoch  899/1000 loss: 0.375151 Accuracy 79.31%
Epoch  999/1000 loss: 0.283293 Accuracy 89.66%
