from torch.utils.data import Dataset, DataLoader
from torch import nn, from_numpy, optim
import numpy as np


# Dataset 을 상속받는 Class
class DiabetesDataset(Dataset): 
    def __init__(self): # Data를 가져오고, Input과 label Data로 나눈다. 또한 Data의 길이도 구한다.
        xy = np.loadtxt('diabetes.csv',delimiter=',', dtype=np.float32) # data load 
        self.len = xy.shape[0] # xy dataset의 길이 
        self.x_data = from_numpy(xy[:, 0:-1]) # data type : ndarray -> tensor 
        self.y_data = from_numpy(xy[:, [-1]]) # data type : ndarray -> tensor 

    def __getitem__(self, index): # Index에 해당하는 데이터 Return 
        return self.x_data[index], self.y_data[index]

    def __len__(self): # Data 의 길이 반환
        return self.len


dataset = DiabetesDataset() # Dataloader객체 생성 : data load, data split, get data length 

train_loader = DataLoader(dataset=dataset, # 배치 크기는 32, Data 섞기, num_workers -> Multiple Process 에서 사용
                          batch_size=32,
                          shuffle=True,
                          num_workers=2)


class Model(nn.Module):  # Wide and Deep 

    def __init__(self): # 3개의 Layer로 구성 
        super(Model, self).__init__()
        self.l1 = nn.Linear(8, 6)
        self.l2 = nn.Linear(6, 4)
        self.l3 = nn.Linear(4, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        out1 = self.sigmoid(self.l1(x))
        out2 = self.sigmoid(self.l2(out1))
        y_pred = self.sigmoid(self.l3(out2))
        return y_pred


model = Model() # Model 객체 생성

criterion = nn.BCELoss(reduction='sum') # Reduction = 'sum' -> Batch 에 대한 Error 를 모두 더한다 
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(2):
    for i, data in enumerate(train_loader, 0): # ##################  0 ?
        inputs, labels = data # train_loader (x_data, y_data) => inputs, labels

        # Forward pass
        y_pred = model(inputs)

        # Compute and print loss
        loss = criterion(y_pred, labels)
        print(f'Epoch {epoch + 1} | Batch: {i+1} | Loss: {loss.item():.4f}')

        # Gradient 초기화
        optimizer.zero_grad()
        # Backpropagation을 통한 Gradient 구하기
        loss.backward()
        # Weight Update
        optimizer.step()
