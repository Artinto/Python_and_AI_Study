import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import sigmoid
from torch import tensor
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# gpu 사용
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 시드 고정
torch.manual_seed(777)
if device == 'cuda' : torch.cuda.manual_seed_all(777)
np.random.seed(777)

dataset_path="/content/diabetes.csv"
dataset=np.loadtxt(dataset_path, delimiter=',',dtype=np.float32)
# train, test data 분리
x_train = torch.tensor(dataset[:660, 0:8], dtype=torch.float32, device=device)
y_train = torch.tensor(dataset[:660, 8:9], dtype=torch.float32, device=device)
ds = TensorDataset(x_train, y_train)
loader = DataLoader(ds, batch_size=66, shuffle=True)
x_test = torch.tensor(dataset[660:,0:8], dtype=torch.float32, device=device)
y_test = torch.tensor(dataset[660:,8:9], dtype=torch.float32, device=device)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(8, 6)
        self.linear2 = nn.Linear(6, 4)
        self.linear3 = nn.Linear(4, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out1 = self.sigmoid(self.linear1(x))
        out2 = self.sigmoid(self.linear2(out1))   # out1의 결과를 out2의 input으로
        y_pred = self.sigmoid(self.linear3(out2)) # out2의 결과를 out3의 input으로
        return y_pred

print("========== Train Start ==========")
lr = 1
while lr >= 1e-4:
    model = Model()
    model.to(device)
    criterion = nn.BCELoss(reduction='mean')
    optimizer = optim.SGD(model.parameters(), lr=lr)
    print("Learning Rate : ", lr)
    cnt = 0
    for epoch in range(4001):
        for x, y in loader:
            y_pred = model(x)
            loss = criterion(y_pred, y)
            #print(f'Epoch {epoch + 1}/1000 | Loss: {loss.item():.4f}')

            if epoch % 1000 == 0:
                for idx in range(len(y)):
                    if(round(y_pred[idx].item()) == y[idx].item()): 
                        cnt+=1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if epoch % 1000 == 0:
            print(f'Loss : {loss.item():.8f} | Accuracy : {cnt / len(y_train) * 100:.8f}')
            cnt = 0   

    with torch.no_grad():
        y_pred = model(x_test)
        for idx in range(len(y_test)):
            if(round(y_pred[idx].item()) == y_test[idx].item()):
                cnt+=1
        print("Test Accuracy : ", cnt / len(y_test) * 100)

    lr /= 10

# ========== Train Start ==========
# Learning Rate :  1
# Loss : 0.69554746 | Accuracy : 65.75757576
# Loss : 0.32509372 | Accuracy : 77.42424242
# Loss : 0.43581358 | Accuracy : 80.60606061
# Loss : 0.40938622 | Accuracy : 83.03030303
# Loss : 0.23617199 | Accuracy : 84.84848485
# Test Accuracy :  64.64646464646465
# Learning Rate :  0.1
# Loss : 0.68282759 | Accuracy : 65.75757576
# Loss : 0.56766868 | Accuracy : 76.21212121
# Loss : 0.33969310 | Accuracy : 76.81818182
# Loss : 0.61289310 | Accuracy : 78.33333333
# Loss : 0.53455538 | Accuracy : 78.78787879
# Test Accuracy :  81.81818181818183
# Learning Rate :  0.01
# Loss : 0.63373005 | Accuracy : 65.75757576
# Loss : 0.54773200 | Accuracy : 65.75757576
# Loss : 0.64490074 | Accuracy : 65.75757576
# Loss : 0.61195731 | Accuracy : 65.75757576
# Loss : 0.61676788 | Accuracy : 65.75757576
# Test Accuracy :  62.62626262626263
# Learning Rate :  0.001
# Loss : 0.64702386 | Accuracy : 65.75757576
# Loss : 0.59672606 | Accuracy : 65.75757576
# Loss : 0.62615573 | Accuracy : 65.75757576
# Loss : 0.65563953 | Accuracy : 65.75757576
# Loss : 0.65552121 | Accuracy : 65.75757576
# Test Accuracy :  62.62626262626263
# Learning Rate :  0.0001
# Loss : 0.73249143 | Accuracy : 34.24242424
# Loss : 0.69618034 | Accuracy : 34.24242424
# Loss : 0.66721696 | Accuracy : 65.75757576
# Loss : 0.64029694 | Accuracy : 65.75757576
# Loss : 0.66393059 | Accuracy : 65.75757576
# Test Accuracy :  62.62626262626263