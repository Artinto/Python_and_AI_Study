import sys
import torch
import torch.nn as nn
from torch.autograd import Variable

torch.manual_seed(777)  # reproducibility
#            0    1    2    3    4
idx2char = ['h', 'i', 'e', 'l', 'o'] #각 문자에 인덱스 부여

# Teach hihell -> ihello
x_data = [0, 1, 0, 2, 3, 3]   # hihell에서 ihello로
one_hot_lookup = [[1, 0, 0, 0, 0],  # 0 one hot으로 표현
                  [0, 1, 0, 0, 0],  # 1
                  [0, 0, 1, 0, 0],  # 2
                  [0, 0, 0, 1, 0],  # 3
                  [0, 0, 0, 0, 1]]  # 4

y_data = [1, 0, 2, 3, 3, 4]    # ihello
x_one_hot = [one_hot_lookup[x] for x in x_data] #one hot으로 x_data표현 

# As we have one batch of samples, we will change them to variables only once
inputs = Variable(torch.Tensor(x_one_hot)) #변환한 x_data와 y_data를 텐서로 변환
labels = Variable(torch.LongTensor(y_data))

num_classes = 5
input_size = 5  # one-hot size
hidden_size = 5  # output from the RNN. 5 to directly predict one-hot
batch_size = 1   # one sentence
sequence_length = 1  # One by one
num_layers = 1  # one-layer rnn


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.rnn = nn.RNN(input_size=input_size,
                          hidden_size=hidden_size, batch_first=True) #인풋5 아웃풋5 input data shape이 (batch_size, seq_len, features(인풋))

    def forward(self, hidden, x): #x와 hidden state를 입력으로 받아서 rnn cell에 넘겨주고 그 결과를 다시 return
        # Reshape input (batch first)
        x = x.view(batch_size, sequence_length, input_size) #(1, 5, 5)형태

        # Propagate input through RNN
        # Input: (batch, seq_len, input_size)
        # hidden: (num_layers * num_directions, batch, hidden_size),(1, 1, 5)형태
        out, hidden = self.rnn(x, hidden) #out과 hidden도출, hidden은 다음 cell의 hidden입력  
        return hidden, out.view(-1, num_classes) #out이 N * 5 shape을 따르게 하기 위해

    def init_hidden(self): #초기엔 hidden이 없기때문에 0으로 초기화 
        # Initialize hidden and cell states
        # (num_layers * num_directions, batch, hidden_size)
        return Variable(torch.zeros(num_layers, batch_size, hidden_size))


# Instantiate RNN model
model = Model()
print(model)

# Set loss and optimizer function
# CrossEntropyLoss = LogSoftmax + NLLLoss
criterion = nn.CrossEntropyLoss() #CELoss 사용
optimizer = torch.optim.Adam(model.parameters(), lr=0.1) #Adam: sgd + momentum, RMSprop을 같이 사용

# Train the model
for epoch in range(100):
    optimizer.zero_grad() #미분값 초기화
    loss = 0
    hidden = model.init_hidden() #zero 벡터로 hidden 초기화

    sys.stdout.write("predicted string: ")
    for input, label in zip(inputs, labels):
        # print(input.size(), label.size())
        hidden, output = model(hidden, input) #model을 통해 hidden, output도출 
        val, idx = output.max(1) #예측값(문자) 저장
        sys.stdout.write(idx2char[idx.data[0]]) #예측값 출력
        loss += criterion(output, torch.LongTensor([label]))

    print(", epoch: %d, loss: %1.3f" % (epoch + 1, loss))

    loss.backward() #역전파
    optimizer.step() #가중치 업데이트

print("Learning finished!")
