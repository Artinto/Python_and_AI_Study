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
        self.linear = nn.Linear(8, 1)

    def forward(self, x):
        y_pred = sigmoid(self.linear(x))
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
    for epoch in range(5001):
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
            print("Accuracy : ", cnt / len(y_train) * 100)
            cnt = 0   

    with torch.no_grad():
        y_pred = model(x_test)
        for idx in range(len(y_test)):
            if(round(y_pred[idx].item()) == y_test[idx].item()):
                cnt+=1
        print("Test Accuracy : ", cnt / len(y_test) * 100)

    lr /= 10