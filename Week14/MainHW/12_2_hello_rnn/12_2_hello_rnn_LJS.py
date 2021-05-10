# Lab 12 RNN
import sys # for prompt command 
import torch
import torch.nn as nn
from torch.autograd import Variable

torch.manual_seed(777)  # reproducibility -> Seed 고정  
#            0    1    2    3    4
idx2char = ['h', 'i', 'e', 'l', 'o']

# Teach hihell -> ihello
x_data = [0, 1, 0, 2, 3, 3]   # hihell

# look-up table
one_hot_lookup = [[1, 0, 0, 0, 0],  # 0
                  [0, 1, 0, 0, 0],  # 1
                  [0, 0, 1, 0, 0],  # 2
                  [0, 0, 0, 1, 0],  # 3
                  [0, 0, 0, 0, 1]]  # 4

# label 
y_data = [1, 0, 2, 3, 3, 4]    # ihello
x_one_hot = [one_hot_lookup[x] for x in x_data]

# x_data 원소에 맞는 one_hot vector 생성

# As we have one batch of samples, we will change them to variables only once -- array -> tensor 로  
inputs = Variable(torch.Tensor(x_one_hot))
labels = Variable(torch.LongTensor(y_data))

num_classes = 5  # 글자 수 
input_size = 5   # one-hot size
hidden_size = 5  # output from the RNN. 5 to directly predict one-hot
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
