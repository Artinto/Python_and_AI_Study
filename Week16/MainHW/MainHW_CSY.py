# 필요한 모듈 import
import numpy as np
import pandas as pd
import pandas_datareader.data as pdr
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable 

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 데이터 읽어오기
df = pd.read_csv("/content/data/data-02-stock_daily_edit.csv")

X = df.drop(columns='Volume')
y = df.iloc[:, 4:5]

print(X)
print(y)

#0~1 범위 변환 함수
# 객체 생성
mm = MinMaxScaler()
ss = StandardScaler()

X_ss = ss.fit_transform(X)
y_mm = mm.fit_transform(y) 


# Train Data
X_train = X_ss[:512, :]
X_test = X_ss[512:, :]

# Test Data 

y_train = y[:512]
y_test = y[512:]
print(y_train)
print(y_test)

# tensor로 변환, Variable()
X_train_tensors = Variable(torch.Tensor(X_train))
X_test_tensors = Variable(torch.Tensor(X_test))

y_train_tensors = Variable(torch.tensor(y_train.values))
y_test_tensors = Variable(torch.tensor(y_test.values))

# reshape
X_train_tensors_final = torch.reshape(X_train_tensors,   (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))
X_test_tensors_final = torch.reshape(X_test_tensors,  (X_test_tensors.shape[0], 1, X_test_tensors.shape[1])) 


print("Training Shape", X_train_tensors_final.shape, y_train_tensors.shape)
print("Testing Shape", X_test_tensors_final.shape, y_test_tensors.shape) 

# device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# torch.nn.Module을 상속받는 파이썬 클래스 정의
# 모델의 생성자 정의 (객체가 갖는 속성값을 초기화하는 역할, 객체가 생성될 때 자동으로 호출됨)
class LSTM1(nn.Module):
  def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
    # super() 함수를 통해 nn.Module의 생성자를 호출함
    # 생성자 정의
    super(LSTM1, self).__init__()
    self.num_classes = num_classes
    self.num_layers = num_layers
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.seq_length = seq_length
 
    self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                      num_layers=num_layers, batch_first=True)
    #fully connected layer
    self.fc_1 =  nn.Linear(hidden_size, 128)
    self.fc = nn.Linear(128, num_classes)

    self.relu = nn.ReLU() 

  def forward(self,x):
    #hidden state
    #internal state
    h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device)
    c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device)   

    #lstm에 input, hidden, and internal state
    output, (hn, cn) = self.lstm(x, (h_0, c_0))
   
    #reshaping the data
    hn = hn.view(-1, self.hidden_size)
    out = self.relu(hn)
    out = self.fc_1(out)
    out = self.relu(out)
    out = self.fc(out)
   
    return out 


num_epochs = 1000
learning_rate = 0.000001

input_size = 4
hidden_size = 1
num_layers = 1

num_classes = 1
lstm1 = LSTM1(num_classes, input_size, hidden_size, num_layers, X_train_tensors_final.shape[1]).to(device)

loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(lstm1.parameters(), lr=learning_rate)



for epoch in range(num_epochs):
  #forward pass
  
  outputs = lstm1.forward(X_train_tensors_final.to(device))
  outputs = outputs.type(torch.FloatTensor)
  # 미분을 통해 얻은 기울기가 이전에 계산된 기울기 값에 누적되기 때문에 기울기 값을 0으로 초기화 함
  optimizer.zero_grad()
  
  y_train_tensors = y_train_tensors.type(torch.FloatTensor)
  # loss function
  loss = loss_function(outputs, y_train_tensors.to(device))

  # 역전파
  loss.backward()

  # step() 함수를 호출하여 parameter를 업데이트함
  optimizer.step()
  if epoch % 10 == 0:
    print("Epoch: %d, loss: %1.5f" % (epoch, loss.item())) 


df_X_ss = ss.transform(df.drop(columns='Volume'))
df_y_mm = mm.transform(df.iloc[:, 4:5])

df_X_ss = Variable(torch.Tensor(df_X_ss))
df_y_mm = Variable(torch.Tensor(df_y_mm))

#reshaping the dataset
df_X_ss = torch.reshape(df_X_ss, (df_X_ss.shape[0], 1, df_X_ss.shape[1]))
train_predict = lstm1(df_X_ss.to(device))
data_predict = train_predict.data.detach().cpu().numpy()
dataY_plot = df_y_mm.data.numpy()

#reverse transformation
data_predict = mm.inverse_transform(data_predict)
dataY_plot = mm.inverse_transform(dataY_plot)
plt.figure(figsize=(10,6))
plt.axvline(x=512, c='r', linestyle='--')

plt.plot(dataY_plot, label='Actuall Data')
plt.plot(data_predict, label='Predicted Data')
plt.title('Time-Series Prediction')
plt.legend()
plt.show()



'''
         # Open        High         Low       Close
0    568.002570  568.002570  552.922516  558.462551
1    561.202549  566.432590  558.672539  559.992565
2    566.892592  567.002574  556.932537  556.972503
3    558.712504  568.452595  558.712504  567.162558
4    599.992707  604.832763  562.192568  567.002574
..          ...         ...         ...         ...
727  819.000000  823.000000  816.000000  820.450012
728  819.359985  823.000000  818.469971  818.979980
729  819.929993  824.400024  818.979980  824.159973
730  823.020020  828.070007  821.655029  828.070007
731  828.659973  833.450012  828.349976  831.659973

[732 rows x 4 columns]
          Close
0    558.462551
1    559.992565
2    556.972503
3    567.162558
4    567.002574
..          ...
727  820.450012
728  818.979980
729  824.159973
730  828.070007
731  831.659973

[732 rows x 1 columns]
          Close
0    558.462551
1    559.992565
2    556.972503
3    567.162558
4    567.002574
..          ...
507  749.909973
508  745.289978
509  737.799988
510  745.690002
511  740.280029

[512 rows x 1 columns]
          Close
512  739.150024
513  736.099976
514  743.090027
515  751.719971
516  753.200012
..          ...
727  820.450012
728  818.979980
729  824.159973
730  828.070007
731  831.659973

[220 rows x 1 columns]
Training Shape torch.Size([512, 1, 4]) torch.Size([512, 1])
Testing Shape torch.Size([220, 1, 4]) torch.Size([220, 1])
Epoch: 0, loss: 364890.78125
Epoch: 10, loss: 364890.34375
Epoch: 20, loss: 364889.93750
Epoch: 30, loss: 364889.50000
Epoch: 40, loss: 364889.09375
Epoch: 50, loss: 364888.68750
Epoch: 60, loss: 364888.28125
Epoch: 70, loss: 364887.84375
Epoch: 80, loss: 364887.43750
Epoch: 90, loss: 364887.00000
Epoch: 100, loss: 364886.59375
Epoch: 110, loss: 364886.18750
Epoch: 120, loss: 364885.75000
Epoch: 130, loss: 364885.34375
Epoch: 140, loss: 364884.93750
Epoch: 150, loss: 364884.50000
Epoch: 160, loss: 364884.09375
Epoch: 170, loss: 364883.68750
Epoch: 180, loss: 364883.25000
Epoch: 190, loss: 364882.84375
Epoch: 200, loss: 364882.43750
Epoch: 210, loss: 364882.00000
Epoch: 220, loss: 364881.59375
Epoch: 230, loss: 364881.18750
Epoch: 240, loss: 364880.75000
Epoch: 250, loss: 364880.34375
Epoch: 260, loss: 364879.93750
Epoch: 270, loss: 364879.50000
Epoch: 280, loss: 364879.09375
Epoch: 290, loss: 364878.68750
Epoch: 300, loss: 364878.25000
Epoch: 310, loss: 364877.84375
Epoch: 320, loss: 364877.43750
Epoch: 330, loss: 364877.00000
Epoch: 340, loss: 364876.59375
Epoch: 350, loss: 364876.18750
Epoch: 360, loss: 364875.75000
Epoch: 370, loss: 364875.34375
Epoch: 380, loss: 364874.96875
Epoch: 390, loss: 364874.50000
Epoch: 400, loss: 364874.09375
Epoch: 410, loss: 364873.68750
Epoch: 420, loss: 364873.25000
Epoch: 430, loss: 364872.84375
Epoch: 440, loss: 364872.40625
Epoch: 450, loss: 364872.03125
Epoch: 460, loss: 364871.59375
Epoch: 470, loss: 364871.15625
Epoch: 480, loss: 364870.75000
Epoch: 490, loss: 364870.34375
Epoch: 500, loss: 364869.90625
Epoch: 510, loss: 364869.50000
Epoch: 520, loss: 364869.09375
Epoch: 530, loss: 364868.65625
Epoch: 540, loss: 364868.25000
Epoch: 550, loss: 364867.84375
Epoch: 560, loss: 364867.40625
Epoch: 570, loss: 364867.06250
Epoch: 580, loss: 364866.59375
Epoch: 590, loss: 364866.15625
Epoch: 600, loss: 364865.78125
Epoch: 610, loss: 364865.34375
Epoch: 620, loss: 364864.90625
Epoch: 630, loss: 364864.53125
Epoch: 640, loss: 364864.09375
Epoch: 650, loss: 364863.65625
Epoch: 660, loss: 364863.28125
Epoch: 670, loss: 364862.84375
Epoch: 680, loss: 364862.40625
Epoch: 690, loss: 364862.00000
Epoch: 700, loss: 364861.59375
Epoch: 710, loss: 364861.15625
Epoch: 720, loss: 364860.75000
Epoch: 730, loss: 364860.34375
Epoch: 740, loss: 364859.90625
Epoch: 750, loss: 364859.46875
Epoch: 760, loss: 364859.09375
Epoch: 770, loss: 364858.65625
Epoch: 780, loss: 364858.21875
Epoch: 790, loss: 364857.84375
Epoch: 800, loss: 364857.40625
Epoch: 810, loss: 364856.96875
Epoch: 820, loss: 364856.59375
Epoch: 830, loss: 364856.15625
Epoch: 840, loss: 364855.71875
Epoch: 850, loss: 364855.34375
Epoch: 860, loss: 364854.90625
Epoch: 870, loss: 364854.46875
Epoch: 880, loss: 364854.09375
Epoch: 890, loss: 364853.65625
Epoch: 900, loss: 364853.21875
Epoch: 910, loss: 364852.84375
Epoch: 920, loss: 364852.40625
Epoch: 930, loss: 364851.93750
Epoch: 940, loss: 364851.56250
Epoch: 950, loss: 364851.15625
Epoch: 960, loss: 364850.68750
Epoch: 970, loss: 364850.34375
Epoch: 980, loss: 364849.90625
Epoch: 990, loss: 364849.43750
'''
