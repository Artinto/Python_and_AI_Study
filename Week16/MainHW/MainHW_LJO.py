import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
 
seq_len = 7
data_dim = 5
hidden_dim = 10
output_dim = 1
learning_rate = 0.01
iterations = 500

xy = np.loadtxt('/content/drive/MyDrive/stock.csv', delimiter=',')
#xy = [::-1]

train_size = int(len(xy) * 0.7)
train_set = xy[0:train_size]
test_set = xy[train_size - seq_len:]

def minmax_scaler(data):
    numerator = data - np.min(data,0)
    denominator = np.max(data,0) - np.min(data,0)
    return numerator / (denominator + 1e-7)

def build_dataset(time_series, seq_len):
    dataX = []
    dataY = []
    for i in range(0, len(time_series) - seq_len):
        _x = time_series[i:i + seq_len, :]
        _y = time_series[i + seq_len, [-1]]
        dataX.append(_x)
        dataY.append(_y)

    return np.array(dataX), np.array(dataY)

train_set = minmax_scaler(train_set)
test_set = minmax_scaler(test_set)

trainX, trainY = build_dataset(train_set, seq_len)
testX, testY = build_dataset(test_set, seq_len) 

trainX_tensor = torch.FloatTensor(trainX)
trainY_tensor = torch.FloatTensor(trainY)

testX_tensor = torch.FloatTensor(testX)
testY_tensor = torch.FloatTensor(testY)

class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers):
        super(Net, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim, bias=True)

    def forward(self, x):
        x, _status = self.rnn(x)
        x = self.fc(x[:, -1])
        return x

net = Net(data_dim, hidden_dim, output_dim, 1)

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

for i in range(iterations):
    optimizer.zero_grad()
    outputs = net(trainX_tensor)
    loss = criterion(outputs, trainY_tensor)
    loss.backward()
    if i % 100 == 0:
      print("Epoch: %d, loss: %1.5f" % (i, loss.item())) 
    optimizer.step()

plt.plot(testY)
plt.plot(net(testX_tensor).data.numpy())
plt.legend(['origin', 'pred'])
plt.show()

Epoch: 0, loss: 0.12275
Epoch: 100, loss: 0.00135
Epoch: 200, loss: 0.00094
Epoch: 300, loss: 0.00075
Epoch: 400, loss: 0.00063
