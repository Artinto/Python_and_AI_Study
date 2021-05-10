import sys
import torch
import torch.nn as nn
from torch.autograd import Variable

torch.manual_seed(777)  # reproducibility
#            0    1    2    3    4
idx2char = ['h', 'i', 'e', 'l', 'o'] 

# Teach hihell -> ihello
x_data = [0, 1, 0, 2, 3, 3]   # hihell
one_hot_lookup = [[1, 0, 0, 0, 0],  # 0 
                  [0, 1, 0, 0, 0],  # 1
                  [0, 0, 1, 0, 0],  # 2
                  [0, 0, 0, 1, 0],  # 3
                  [0, 0, 0, 0, 1]]  # 4

y_data = [1, 0, 2, 3, 3, 4]    # ihello #원하는 결과를 미리 정해주고 예측
x_one_hot = [one_hot_lookup[x] for x in x_data]

# As we have one batch of samples, we will change them to variables only once
inputs = Variable(torch.Tensor(x_one_hot))
labels = Variable(torch.LongTensor(y_data))

num_classes = 5
input_size = 5  # one-hot size 인풋 사이즈
hidden_size = 5  # output from the RNN. 5 to directly predict one-hot 아웃풋 사이즈
batch_size = 1   # one sentence 배치 사이즈
sequence_length = 1  # One by one 시퀀스 사이즈
num_layers = 1  # one-layer rnn 레이어 사이즈


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.rnn = nn.RNN(input_size=input_size,
                          hidden_size=hidden_size, batch_first=True) #뉴런 생성

    def forward(self, hidden, x): #강의에서는 x와 hidden의 자리가 다름 그러나 큰 의미 없음
        # Reshape input (batch first)
        x = x.view(batch_size, sequence_length, input_size)

        # Propagate input through RNN
        # Input: (batch, seq_len, input_size)
        # hidden: (num_layers * num_directions, batch, hidden_size)
        out, hidden = self.rnn(x, hidden) #rnn계산
        return hidden, out.view(-1, num_classes) #아웃풋을 num_classes로 맞춰야 계산이 가능

    def init_hidden(self):
        # Initialize hidden and cell states
        # (num_layers * num_directions, batch, hidden_size)
        return Variable(torch.zeros(num_layers, batch_size, hidden_size))


# Instantiate RNN model
model = Model()
print(model)

# Set loss and optimizer function
# CrossEntropyLoss = LogSoftmax + NLLLoss
criterion = nn.CrossEntropyLoss() #로스 계산
optimizer = torch.optim.Adam(model.parameters(), lr=0.1) #로스 계산해주는 계수들 
 
# Train the model
for epoch in range(100):
    optimizer.zero_grad()
    loss = 0
    hidden = model.init_hidden()

    sys.stdout.write("predicted string: ")
    for input, label in zip(inputs, labels): #1 input, 1 label
        # print(input.size(), label.size())
        hidden, output = model(hidden, input) #???? 모델을 거치면 input이 히든이 되어야하고, hidden이 output이 되어야함.
        val, idx = output.max(1)
        sys.stdout.write(idx2char[idx.data[0]])
        loss += criterion(output, torch.LongTensor([label])) # 코드의 아웃풋과 내가 입력한 결과를 비교하여 로스 측정

    print(", epoch: %d, loss: %1.3f" % (epoch + 1, loss))

    loss.backward()
    optimizer.step()

print("Learning finished!")
