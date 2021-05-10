# Lab 12 RNN
import sys # for prompt command 
import torch
import torch.nn as nn
from torch.autograd import Variable

torch.manual_seed(777)  # reproducibility -> Seed 고정  
#            0    1    2    3    4
idx2char = ['h', 'i', 'e', 'l', 'o','w','r','d']

# Teach hihell -> ihelloworld
x_data = [7, 1, 0, 2, 3, 3, 4, 5, 4, 6, 3]   # dihelloworl

# look-up table
one_hot_lookup = [[1, 0, 0, 0, 0, 0 ,0 ,0],  # 0
                  [0, 1, 0, 0, 0, 0, 0, 0],  # 1
                  [0, 0, 1, 0, 0, 0, 0, 0],  # 2
                  [0, 0, 0, 1, 0, 0, 0, 0],  # 3
                  [0, 0, 0, 0, 1, 0, 0, 0],  # 4
                  [0, 0, 0, 0, 0, 1, 0, 0],  # 5
                  [0, 0, 0, 0, 0, 0, 1, 0],  # 6
                  [0, 0, 0, 0, 0, 0, 0, 1]]  # 7

# label 
y_data = [1, 0, 2, 3, 3, 4, 5, 4, 6, 3, 7]    # ihelloworld
x_one_hot = [one_hot_lookup[x] for x in x_data]

print(x_one_hot)

# x_data 원소에 맞는 one_hot vector 생성

# As we have one batch of samples, we will change them to variables only once -- array -> tensor 로  
inputs = Variable(torch.Tensor(x_one_hot))
labels = Variable(torch.LongTensor(y_data))

num_classes = 8  # 글자 수 
input_size = 8   # one-hot size
hidden_size = 8  # output from the RNN. 5 to directly predict one-hot
batch_size = 1   # one sentence
sequence_length = 1  # One by one
num_layers = 1  # one-layer rnn


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        # nn.RNN(input_size=5, hidden_size=5, batch_first=True)
        self.rnn = nn.RNN(input_size=input_size,
                          hidden_size=hidden_size, batch_first=True)

    def forward(self, hidden, x):
        # Reshape input (batch first)
        x = x.view(batch_size, sequence_length, input_size)

        # Propagate input through RNN
        # Input: (batch, seq_len, input_size)
        # hidden: (num_layers * num_directions, batch, hidden_size)
        out, hidden = self.rnn(x, hidden)  # x : input 
        return hidden, out.view(-1, num_classes) # hidden output 다시 사용하기 위해 Return # 모델 Output은 batch size x 글자 수 shape으로 변형 

    def init_hidden(self):
        # Initialize hidden and cell states
        # (num_layers * num_directions, batch, hidden_size)
        return Variable(torch.zeros(num_layers, batch_size, hidden_size))


# Instantiate RNN model
model = Model()
print(model)

# Set loss and optimizer function
# CrossEntropyLoss = LogSoftmax + NLLLoss
criterion = nn.CrossEntropyLoss() # Define loss function 
optimizer = torch.optim.Adam(model.parameters(), lr=0.1) # Define gradient update function 

# Train the model
for epoch in range(100):
    optimizer.zero_grad() # Gradient initialize 
    loss = 0
    hidden = model.init_hidden() # Tensor Variable 선언 

    sys.stdout.write("predicted string: ")
    
    for input, label in zip(inputs, labels):
        # print(input.size(), label.size())
        hidden, output = model(hidden, input) # input -> (batch_size , seq_num, input_size) shape로 변형 후 Rnn 모델로 
        val, idx = output.max(1) # output은 softmax 거쳐서 나옴 -> max value & index return 
        #index에 해당 하는 글자 출력
        sys.stdout.write(idx2char[idx.data[0]]) # idx : tensor기 때문에 idx.data[0]과 같은 형태로 사용 
        loss += criterion(output, torch.LongTensor([label]))

    print(", epoch: %d, loss: %1.3f" % (epoch + 1, loss))

    loss.backward()
    optimizer.step()

print("Learning finished!")



'''
predicted string: idedddddddd, epoch: 1, loss: 24.352
predicted string: ihidlddihid, epoch: 2, loss: 20.858
predicted string: ihirlddihlo, epoch: 3, loss: 18.858
predicted string: ihirlooirld, epoch: 4, loss: 17.218
predicted string: iherlowirld, epoch: 5, loss: 15.962
predicted string: ihireoworld, epoch: 6, loss: 14.909
predicted string: ihirloworld, epoch: 7, loss: 13.981
predicted string: ihirloworld, epoch: 8, loss: 13.222
predicted string: ihirloworld, epoch: 9, loss: 12.688
predicted string: ihirldworld, epoch: 10, loss: 12.207
predicted string: ihirldworld, epoch: 11, loss: 11.758
predicted string: ihirldworld, epoch: 12, loss: 11.375
predicted string: ihirloworld, epoch: 13, loss: 11.048
predicted string: iherloworld, epoch: 14, loss: 10.764
predicted string: iherloworld, epoch: 15, loss: 10.521
predicted string: iherloworld, epoch: 16, loss: 10.308
predicted string: iherloworld, epoch: 17, loss: 10.113
predicted string: iherloworld, epoch: 18, loss: 9.919
predicted string: iherloworld, epoch: 19, loss: 9.711
predicted string: iherloworld, epoch: 20, loss: 9.553
predicted string: iherloworld, epoch: 21, loss: 9.406
predicted string: iherloworld, epoch: 22, loss: 9.280
predicted string: iherloworld, epoch: 23, loss: 9.165
predicted string: iherloworld, epoch: 24, loss: 9.063
predicted string: iherloworld, epoch: 25, loss: 8.955
predicted string: iherloworld, epoch: 26, loss: 8.860
predicted string: iherloworld, epoch: 27, loss: 8.777
predicted string: ihelloworld, epoch: 28, loss: 8.693
predicted string: ihelloworld, epoch: 29, loss: 8.618
predicted string: ihelloworld, epoch: 30, loss: 8.558
predicted string: ihelloworld, epoch: 31, loss: 8.495
predicted string: ihelloworld, epoch: 32, loss: 8.439
predicted string: ihelloworld, epoch: 33, loss: 8.398
predicted string: ihelloworld, epoch: 34, loss: 8.343
predicted string: ihelloworld, epoch: 35, loss: 8.312
predicted string: ihelloworld, epoch: 36, loss: 8.267
predicted string: ihelloworld, epoch: 37, loss: 8.249
predicted string: ihelloworld, epoch: 38, loss: 8.246
predicted string: ihelloworld, epoch: 39, loss: 8.185
predicted string: ihelloworld, epoch: 40, loss: 8.213
predicted string: ihelloworld, epoch: 41, loss: 8.227
predicted string: ihelloworld, epoch: 42, loss: 8.185
predicted string: ihelloworld, epoch: 43, loss: 8.151
predicted string: ihelloworld, epoch: 44, loss: 8.170
predicted string: ihelloworld, epoch: 45, loss: 8.117
predicted string: ihelloworld, epoch: 46, loss: 8.159
predicted string: ihelloworld, epoch: 47, loss: 8.094
predicted string: ihelloworld, epoch: 48, loss: 8.107
predicted string: ihelloworld, epoch: 49, loss: 8.055
predicted string: ihelloworld, epoch: 50, loss: 8.072
predicted string: ihelloworld, epoch: 51, loss: 8.044
predicted string: ihelloworld, epoch: 52, loss: 8.043
predicted string: ihelloworld, epoch: 53, loss: 8.018
predicted string: ihelloworld, epoch: 54, loss: 8.022
predicted string: ihelloworld, epoch: 55, loss: 8.007
predicted string: ihelloworld, epoch: 56, loss: 8.003
predicted string: ihelloworld, epoch: 57, loss: 7.989
predicted string: ihelloworld, epoch: 58, loss: 7.990
predicted string: ihelloworld, epoch: 59, loss: 7.975
predicted string: ihelloworld, epoch: 60, loss: 7.982
predicted string: ihelloworld, epoch: 61, loss: 7.994
predicted string: ihelloworld, epoch: 62, loss: 7.959
predicted string: ihelloworld, epoch: 63, loss: 7.981
predicted string: ihelloworld, epoch: 64, loss: 8.019
predicted string: ihelloworld, epoch: 65, loss: 7.959
predicted string: ihelloworld, epoch: 66, loss: 8.028
predicted string: ihelloworld, epoch: 67, loss: 8.020
predicted string: ihelloworld, epoch: 68, loss: 8.059
predicted string: ihelloworld, epoch: 69, loss: 7.959
predicted string: ihelloworld, epoch: 70, loss: 8.103
predicted string: ihelloworld, epoch: 71, loss: 8.018
predicted string: ihelloworld, epoch: 72, loss: 8.216
predicted string: ihelloworld, epoch: 73, loss: 8.052
predicted string: ihelloworld, epoch: 74, loss: 7.985
predicted string: ihelloworld, epoch: 75, loss: 8.141
predicted string: ihelloworld, epoch: 76, loss: 7.980
predicted string: ihelloworld, epoch: 77, loss: 8.025
predicted string: ihelloworld, epoch: 78, loss: 8.102
predicted string: ihelloworld, epoch: 79, loss: 8.026
predicted string: ihelloworld, epoch: 80, loss: 7.977
predicted string: ihelloworld, epoch: 81, loss: 7.979
predicted string: ihelloworld, epoch: 82, loss: 8.048
predicted string: ihelloworld, epoch: 83, loss: 7.978
predicted string: ihelloworld, epoch: 84, loss: 7.962
predicted string: ihelloworld, epoch: 85, loss: 7.973
predicted string: ihelloworld, epoch: 86, loss: 7.996
predicted string: ihelloworld, epoch: 87, loss: 7.969
predicted string: ihelloworld, epoch: 88, loss: 7.934
predicted string: ihelloworld, epoch: 89, loss: 7.946
predicted string: ihelloworld, epoch: 90, loss: 7.964
predicted string: ihelloworld, epoch: 91, loss: 7.923
predicted string: ihelloworld, epoch: 92, loss: 7.927
predicted string: ihelloworld, epoch: 93, loss: 7.945
predicted string: ihelloworld, epoch: 94, loss: 7.928
predicted string: ihelloworld, epoch: 95, loss: 7.907
predicted string: ihelloworld, epoch: 96, loss: 7.916
predicted string: ihelloworld, epoch: 97, loss: 7.918
predicted string: ihelloworld, epoch: 98, loss: 7.894
predicted string: ihelloworld, epoch: 99, loss: 7.905
predicted string: ihelloworld, epoch: 100, loss: 7.905
Learning finished!
'''
