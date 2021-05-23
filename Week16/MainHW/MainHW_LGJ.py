import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

stock = pd.read_csv('C:\\Users\\GIJIN LEE\\Desktop\\jusik\\stock.csv', sep=',')

scaler = MinMaxScaler()
stock[['Open', 'High', 'Low', 'Volume', 'Close']] = scaler.fit_transform(stock[['Open', 'High', 'Low', 'Volume', 'Close']])
print(stock.head())

print(stock.info())

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'{device} is available')

datax = stock[['Open','High','Low','Volume']].values
result = stock['Close'].values

def seq_data(datax, result, sequence_length):
    datax_seq = []
    result_seq = []
    for i in range(len(datax) - sequence_length):
        datax_seq.append(datax[i: i + sequence_length])
        result_seq.append(result[i + sequence_length])

    return torch.FloatTensor(datax_seq).to(device), torch.FloatTensor(result_seq).to(device).view([-1, 1])  # float형 tensor로 변형, gpu사용가능하게 .to(device)를 사용.

split = 513
sequence_length = 14

datax_seq, result_seq = seq_data(datax, result, sequence_length)

datax_stock_seq = datax_seq[:split]
result_stock_seq = result_seq[:split]
datax_test_seq = datax_seq[split:]
result_test_seq = result_seq[split:]
print(datax_stock_seq.size(), result_stock_seq.size())
print(datax_test_seq.size(), result_test_seq.size())

train = torch.utils.data.TensorDataset(datax_stock_seq, result_stock_seq)
test = torch.utils.data.TensorDataset(datax_test_seq, result_test_seq)

batch_size = 16
train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=batch_size, shuffle=False)

input_size = datax_seq.size(2)
num_layers = 2
hidden_size = 8

class VanillaRNN(nn.Module):

  def __init__(self, input_size, hidden_size, sequence_length, num_layers, device):
    super(VanillaRNN, self).__init__()
    self.device = device
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
    self.fc = nn.Sequential(nn.Linear(hidden_size * sequence_length, 1), nn.Sigmoid())

  def forward(self, x):
    h0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size).to(self.device) # 초기 hidden state 설정하기.
    out, _ = self.rnn(x, h0) # out: RNN의 마지막 레이어로부터 나온 output feature 를 반환한다. hn: hidden state를 반환한다.
    out = out.reshape(out.shape[0], -1) # many to many 전략
    out = self.fc(out)
    return out

model = VanillaRNN(input_size=input_size,
                   hidden_size=hidden_size,
                   sequence_length=sequence_length,
                   num_layers=num_layers,
                   device=device).to(device)

loss_graph = [] # 그래프 그릴 목적인 loss.
n = len(train_loader)

criterion = nn.MSELoss()

lr = 1e-3
num_epochs = 200
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(num_epochs):
  running_loss = 0.0

  for data in train_loader:

    seq, target = data # 배치 데이터.
    out = model(seq)   # 모델에 넣고,
    loss = criterion(out, target) # output 가지고 loss 구하고,

    optimizer.zero_grad() #
    loss.backward() # loss가 최소가 되게하는
    optimizer.step() # 가중치 업데이트 해주고,
    running_loss += loss.item() # 한 배치의 loss 더해주고,

  loss_graph.append(running_loss / n) # 한 epoch에 모든 배치들에 대한 평균 loss 리스트에 담고,
  if epoch % 10 == 0:
    print('[epoch: %d] loss: %.4f'%(epoch, running_loss/n))

plt.figure(figsize=(20,10))
plt.plot(loss_graph)
plt.show()


def plotting(train_loader, test_loader, actual):
    with torch.no_grad():
        train_pred = []
        test_pred = []

        for data in train_loader:
            seq, target = data
            out = model(seq)
            train_pred += out.cpu().numpy().tolist()

        for data in test_loader:
            seq, target = data
            out = model(seq)
            test_pred += out.cpu().numpy().tolist()

    total = train_pred + test_pred
    plt.figure(figsize=(20, 10))
    plt.plot(np.ones(100) * len(train_pred), np.linspace(0, 1, 100), '--', linewidth=0.6)
    plt.plot(actual, '--')
    plt.plot(total, 'b', linewidth=0.6)

    plt.legend(['train boundary', 'actual', 'prediction'])
    plt.show()

plotting(train_loader, test_loader, stock['Close'][sequence_length:])




"C:\Users\GIJIN LEE\anaconda3\envs\pytorch\python.exe" C:/source/PythonAI/data/python.py
       Open      High       Low    Volume     Close
0  0.973336  0.975432  1.000000  0.111123  0.988313
1  0.956900  0.959881  0.980354  0.142502  0.977850
2  0.947896  0.949273  0.972505  0.114170  0.966455
3  0.946235  0.945227  0.971008  0.116169  0.951358
4  0.945186  0.945227  0.963761  0.093726  0.955642
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 732 entries, 0 to 731
Data columns (total 5 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   Open    732 non-null    float64
 1   High    732 non-null    float64
 2   Low     732 non-null    float64
 3   Volume  732 non-null    float64
 4   Close   732 non-null    float64
dtypes: float64(5)
memory usage: 28.7 KB
None
cuda:0 is available
torch.Size([513, 14, 4]) torch.Size([513, 1])
torch.Size([205, 14, 4]) torch.Size([205, 1])
[epoch: 0] loss: 0.1018
[epoch: 10] loss: 0.0055
[epoch: 20] loss: 0.0044
[epoch: 30] loss: 0.0038
[epoch: 40] loss: 0.0033
[epoch: 50] loss: 0.0030
[epoch: 60] loss: 0.0026
[epoch: 70] loss: 0.0021
[epoch: 80] loss: 0.0017
[epoch: 90] loss: 0.0016
[epoch: 100] loss: 0.0014
[epoch: 110] loss: 0.0013
[epoch: 120] loss: 0.0012
[epoch: 130] loss: 0.0010
[epoch: 140] loss: 0.0010
[epoch: 150] loss: 0.0009
[epoch: 160] loss: 0.0008
[epoch: 170] loss: 0.0008
[epoch: 180] loss: 0.0008
[epoch: 190] loss: 0.0008

Process finished with exit code 0
