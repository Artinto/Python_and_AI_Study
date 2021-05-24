import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler #정규화
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt #plot용

# 데이터 읽어오기
stock = pd.read_csv("stock.csv")
scaler = MinMaxScaler()

stock[['Open', 'High', 'Low', 'Volume', 'Close']] = scaler.fit_transform(stock[['Open', 'High', 'Low', 'Volume', 'Close']])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

x = stock[['Open','High','Low','Volume']].values #dataset들을 목표들을 기준으로 분류
y = stock['Close'].values

def seq_data(datax, result, sequence_length): #sequence_length는 몇일동안의 데이터를 이용할 건지
    datax_seq = [] # open, high, low,volume
    result_seq = [] #Close
    for i in range(len(datax) - sequence_length):
        datax_seq.append(datax[i: i + sequence_length])
        result_seq.append(result[i + sequence_length])

    return torch.FloatTensor(datax_seq).to(device), torch.FloatTensor(result_seq).view([-1, 1])

split = 512 # 70퍼센트
sequence_length = 14 #14일간의 데이터

datax_seq, result_seq = seq_data(x, y, sequence_length)
print(datax_seq)
print(result_seq)

datax_stock_seq = datax_seq[:split]
result_stock_seq = result_seq[:split]

datax_test_seq = datax_seq[split:]
result_test_seq = result_seq[split:]

print("Train사이즈")
print(datax_stock_seq.size(), result_stock_seq.size())

print("Test사이즈")
print(datax_test_seq.size(), result_test_seq.size())

train = torch.utils.data.TensorDataset(datax_stock_seq, result_stock_seq)
test = torch.utils.data.TensorDataset(datax_test_seq, result_test_seq)

batch_size = 20
train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=False) #
test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=batch_size, shuffle=False)
#여기까지 기초설정

input_size = datax_seq.size(2)
num_layers = 1 #최종출력
hidden_size = 8 #크기

class VanillaRNN(nn.Module): #이걸 쓸거임.

  def __init__(self, input_size, hidden_size, sequence_length, num_layers, device):
    super(VanillaRNN, self).__init__()
    self.device = device
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
    self.fc = nn.Sequential(nn.Linear(hidden_size * sequence_length, 1), nn.Sigmoid()) #sequential은 moduel을 한번에 돌려버리는 방법 간단한 모델을 구현시에 유용

  def forward(self, x):
    h0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size).to(self.device) # 초기화 및 hidden sate를 저장
    out, _ = self.rnn(x, h0)
    out = out.reshape(out.shape[0], -1) # many to many
    out = self.fc(out)
    return out

model = VanillaRNN(input_size=input_size,
                   hidden_size=hidden_size,
                   sequence_length=sequence_length,
                   num_layers=num_layers,
                   device=device).to(device)

criterion = nn.MSELoss()

lr = 0.001
num_epochs = 800
optimizer = optim.Adam(model.parameters(), lr=lr)

loss_graph = [] # 그래프 그릴 목적인 loss.
n = len(train_loader)

for epoch in range(num_epochs):
  running_loss = 0.0

  for data in train_loader:

    seq, target = data #train용 변수
    out = model(seq) #output
    loss = criterion(out, target)

    optimizer.zero_grad() #
    loss.backward()
    optimizer.step()
    running_loss += loss.item()

  if epoch % 100 == 0:
    print('[epoch: %d] loss: %.4f'%(epoch, running_loss/n))


#그래프
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
