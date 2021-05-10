import torch
import torch.nn as nn
from torch.autograd import Variable

# One hot encoding for each char in 'hello'
h = [1, 0, 0, 0] #0 각각의 문자에 인덱스 부여
e = [0, 1, 0, 0] #1
l = [0, 0, 1, 0] #2
o = [0, 0, 0, 1] #3

# One cell RNN input_dim (4) -> output_dim (2). sequence: 5
cell = nn.RNN(input_size=4, hidden_size=2, batch_first=True) #인풋 4, 아웃풋 2, batch_first는 input data shape에 따라 결정

# (num_layers * num_directions, batch, hidden_size) whether batch_first=True or False
hidden = Variable(torch.randn(1, 1, 2)) #randn은 평균이 0이고 표준편차가 1인 정규분포 생성

# Propagate input through RNN
# Input: (batch, seq_len, input_size) when batch_first=True
inputs = Variable(torch.Tensor([h, e, l, l, o])) #input data shape이 (batch_size, seq_len, features(인풋))이면 batch_first=True
for one in inputs:
    one = one.view(1, 1, -1) #inputs안의 문자들을 [[]]로 감쌈
    # Input: (batch, seq_len, input_size) when batch_first=True
    out, hidden = cell(one, hidden) #초기 히든과 문자로 out을 도출하고 히든은 다음 셀의 히든으로 들어감
    print("one input size", one.size(), "out size", out.size())

# We can do the whole at once
# Propagate input through RNN
# Input: (batch, seq_len, input_size) when batch_first=True
#input data shape이 (batch_size, seq_len, features)라면 batch_first=True
inputs = inputs.view(1, 5, -1) #시퀀스 5번, 주어진 문자가 h,e,l,o이므로 (1,5,4)형태
out, hidden = cell(inputs, hidden)
print("sequence input size", inputs.size(), "out size", out.size())


# hidden : (num_layers * num_directions, batch, hidden_size) whether batch_first=True or False
hidden = Variable(torch.randn(1, 3, 2)) #배치사이즈3,한번에 3배치씩 처리

# One cell RNN input_dim (4) -> output_dim (2). sequence: 5, batch 3
# 3 batches 'hello', 'eolll', 'lleel'
# rank = (3, 5, 4)
inputs = Variable(torch.Tensor([[h, e, l, l, o], #3개의 배치사이즈 시퀀스 5번, 인풋 사이즈
                                [e, o, l, l, l],
                                [l, l, e, e, l]]))

# Propagate input through RNN
# Input: (batch, seq_len, input_size) when batch_first=True
# B x S x I
out, hidden = cell(inputs, hidden)
print("batch input size", inputs.size(), "out size", out.size())


# One cell RNN input_dim (4) -> output_dim (2)
cell = nn.RNN(input_size=4, hidden_size=2) #inputs.transpose(dim0=0, dim1=1)으로
                                           #input data shape이 (seq_len, batch_size, features(인풋))이기 때문에
                                           #batch_first=True를 사용X

# The given dimensions dim0 and dim1 are swapped.
inputs = inputs.transpose(dim0=0, dim1=1)
# Propagate input through RNN
# Input: (seq_len, batch_size, input_size) when batch_first=False (default)
# S x B x I
out, hidden = cell(inputs, hidden)
print("batch input size", inputs.size(), "out size", out.size())
