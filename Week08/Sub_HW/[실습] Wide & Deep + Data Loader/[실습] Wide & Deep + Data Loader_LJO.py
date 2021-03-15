from torch.utils.data import Dataset, DataLoader
from torch import tensor, nn, sigmoid, from_numpy
from sklearn.model_selection import train_test_split
import torch.optim as optim
import numpy as np
import torch

xy = np.loadtxt('/content/drive/MyDrive/diabetes.csv', delimiter=',', dtype=np.float32)
x_data = from_numpy(xy[:, 0:-1])
y_data = from_numpy(xy[:, [-1]])
x_train, x_test, y_train, y_test = train_test_split(x_data,y_data,shuffle=False)
#print("x_train : ",x_train.shape[0],"x_test : ", x_test.shape[0],"y_train : ", y_train.shape[0],"y_test : ", y_test.shape[0])
class DiabetesDataset(Dataset):
    def __init__(self):
        #xy = np.loadtxt('/content/drive/MyDrive/diabetes.csv', delimiter=',', dtype=np.float32)
        self.len = x_train.shape[0]
        self.x_train = x_train
        self.y_train = y_train
    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index]
    def __len__(self):
        return self.len

dataset = DiabetesDataset()
train_loader = DataLoader(dataset=dataset, batch_size=24, shuffle=True, num_workers=0)   
count=0
cnt=0

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = nn.Linear(8, 6)
        self.l2 = nn.Linear(6, 4)
        self.l3 = nn.Linear(4, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out1 = sigmoid(self.l1(x))
        out2 = sigmoid(self.l2(out1))
        y_pred = sigmoid(self.l3(out2))
        return y_pred

model = Model()
criterion = nn.BCELoss(reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=0.35)

for epoch in range(300):
    if (epoch+1)%100==0:
        print("Epoch :",epoch+1)
    for i, data in enumerate(train_loader, 0):
        cnt+=1
        cnt1=0
        inputs, labels = data
        inputs, labels = tensor(inputs), tensor(labels)
        y_pred = model(inputs)
        loss = criterion(y_pred, labels)
        if (epoch+1)%100==0:
            y_pred=y_pred.squeeze(1)
            labels=labels.squeeze(1)
            y_pred=torch.round(y_pred)
            for i in range(len(labels)):
                if y_pred[i]==labels[i]:
                    cnt1+=1
            print(f'batch {cnt} | Loss: {loss.item():.4f} | accuracy {(cnt1/len(labels)*100):.4f}%')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    cnt=0
    cnt1=0
    
y_pred=model(x_test)
y_pred=y_pred.squeeze(1)
y_test=y_test.squeeze(1)
y_pred=torch.round(y_pred)
for i in range(len(y_test)):
    if y_pred[i]==y_test[i]:
        count+=1
print("accuracy : ",(count/len(y_test))*100,"%")

#결과
Epoch : 100
batch 1 | Loss: 0.3745 | accuracy 83.3333%
batch 2 | Loss: 0.6209 | accuracy 75.0000%
batch 3 | Loss: 0.4430 | accuracy 79.1667%
batch 4 | Loss: 0.5016 | accuracy 66.6667%
batch 5 | Loss: 0.3859 | accuracy 79.1667%
batch 6 | Loss: 0.2852 | accuracy 83.3333%
batch 7 | Loss: 0.6795 | accuracy 70.8333%
batch 8 | Loss: 0.4010 | accuracy 75.0000%
batch 9 | Loss: 0.6357 | accuracy 58.3333%
batch 10 | Loss: 0.5218 | accuracy 70.8333%
batch 11 | Loss: 0.4096 | accuracy 83.3333%
batch 12 | Loss: 0.6561 | accuracy 70.8333%
batch 13 | Loss: 0.4064 | accuracy 83.3333%
batch 14 | Loss: 0.3154 | accuracy 87.5000%
batch 15 | Loss: 0.5680 | accuracy 66.6667%
batch 16 | Loss: 0.3668 | accuracy 87.5000%
batch 17 | Loss: 0.6167 | accuracy 66.6667%
batch 18 | Loss: 0.4199 | accuracy 79.1667%
batch 19 | Loss: 0.6811 | accuracy 62.5000%
batch 20 | Loss: 0.5053 | accuracy 66.6667%
batch 21 | Loss: 0.4815 | accuracy 66.6667%
batch 22 | Loss: 0.4780 | accuracy 79.1667%
batch 23 | Loss: 0.3792 | accuracy 87.5000%
batch 24 | Loss: 0.2583 | accuracy 94.1176%
Epoch : 200
batch 1 | Loss: 0.4582 | accuracy 79.1667%
batch 2 | Loss: 0.5956 | accuracy 70.8333%
batch 3 | Loss: 0.4387 | accuracy 75.0000%
batch 4 | Loss: 0.4170 | accuracy 87.5000%
batch 5 | Loss: 0.4009 | accuracy 75.0000%
batch 6 | Loss: 0.4715 | accuracy 75.0000%
batch 7 | Loss: 0.2952 | accuracy 91.6667%
batch 8 | Loss: 0.4751 | accuracy 75.0000%
batch 9 | Loss: 0.4427 | accuracy 75.0000%
batch 10 | Loss: 0.3659 | accuracy 91.6667%
batch 11 | Loss: 0.3609 | accuracy 83.3333%
batch 12 | Loss: 0.4962 | accuracy 75.0000%
batch 13 | Loss: 0.4701 | accuracy 75.0000%
batch 14 | Loss: 0.6901 | accuracy 62.5000%
batch 15 | Loss: 0.5988 | accuracy 70.8333%
batch 16 | Loss: 0.4116 | accuracy 87.5000%
batch 17 | Loss: 0.4300 | accuracy 79.1667%
batch 18 | Loss: 0.4289 | accuracy 75.0000%
batch 19 | Loss: 0.4712 | accuracy 75.0000%
batch 20 | Loss: 0.5176 | accuracy 70.8333%
batch 21 | Loss: 0.6383 | accuracy 66.6667%
batch 22 | Loss: 0.4571 | accuracy 75.0000%
batch 23 | Loss: 0.4174 | accuracy 75.0000%
batch 24 | Loss: 0.4454 | accuracy 82.3529%
Epoch : 300
batch 1 | Loss: 0.4975 | accuracy 70.8333%
batch 2 | Loss: 0.3800 | accuracy 83.3333%
batch 3 | Loss: 0.3336 | accuracy 91.6667%
batch 4 | Loss: 0.6492 | accuracy 54.1667%
batch 5 | Loss: 0.6082 | accuracy 70.8333%
batch 6 | Loss: 0.3289 | accuracy 83.3333%
batch 7 | Loss: 0.3507 | accuracy 83.3333%
batch 8 | Loss: 0.5235 | accuracy 75.0000%
batch 9 | Loss: 0.3665 | accuracy 79.1667%
batch 10 | Loss: 0.3780 | accuracy 83.3333%
batch 11 | Loss: 0.3764 | accuracy 83.3333%
batch 12 | Loss: 0.2778 | accuracy 87.5000%
batch 13 | Loss: 0.3251 | accuracy 83.3333%
batch 14 | Loss: 0.5140 | accuracy 79.1667%
batch 15 | Loss: 0.6756 | accuracy 62.5000%
batch 16 | Loss: 0.4765 | accuracy 79.1667%
batch 17 | Loss: 0.2656 | accuracy 87.5000%
batch 18 | Loss: 0.5120 | accuracy 79.1667%
batch 19 | Loss: 0.5114 | accuracy 66.6667%
batch 20 | Loss: 0.5075 | accuracy 79.1667%
batch 21 | Loss: 0.4984 | accuracy 79.1667%
batch 22 | Loss: 0.4605 | accuracy 83.3333%
batch 23 | Loss: 0.5009 | accuracy 66.6667%
batch 24 | Loss: 0.5909 | accuracy 64.7059%
accuracy :  82.10526315789474 %
