import torch
import torch.nn as nn
from torch.autograd import Variable  
# autograd : 자동미분 연산 제공 
# autograd.Variable : backward / grad / grad_fn .. -> 대부분의 Tensor 연산 제공 

# One hot encoding for each char in 'hello'
h = [1, 0, 0, 0] # 0
e = [0, 1, 0, 0] # 1
l = [0, 0, 1, 0] # 2
o = [0, 0, 0, 1] # 3

# One cell RNN input_dim (4) -> output_dim (2). sequence: 5
# batch_first : batch argument를 먼저 받겠다는 선언 
cell = nn.RNN(input_size=4, hidden_size=2, batch_first=True)

#-----------------------------------batch size 1--------------------------------------#

# (num_layers * num_directions, batch, hidden_size) whether batch_first=True or False
# Input -> Output 사이의 layer , 계속 Recursive하게 사용
hidden = Variable(torch.randn(1, 1, 2))

'''

inputs = Variable(torch.Tensor([h, e, l, l, o])) == Variable(torch.Tensor( [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1]] )) 

- batch = 1 
- seq_len = 5
- input_size = 4

h = [1, 0, 0, 0] # 0
e = [0, 1, 0, 0] # 1
l = [0, 0, 1, 0] # 2
o = [0, 0, 0, 1] # 3

'''

# Propagate input through RNN -- one by one 
# Input: (batch, seq_len, input_size) when batch_first=True
inputs = Variable(torch.Tensor([h, e, l, l, o])) # (1 , 5 , 4)

# 'h' 'e' 'l' 'o' 각 tensor에 대해 (1 , 5 , 4) -> (1 , 1 , 4) //  ex ) h -- [[[1, 0, 0, 0 ]]]
for one in inputs: 
    one = one.view(1, 1, -1)
    # Input: (batch, seq_len, input_size) when batch_first=True
    out, hidden = cell(one, hidden) 
    print("one input size", one.size(), "out size", out.size()) 
    # one input size torch.Size([1, 1, 4]) out size torch.Size([1, 1, 2])
     

# We can do the whole at once 
# Propagate input through RNN -- at once
# Input: (batch, seq_len, input_size) when batch_first=True
inputs = inputs.view(1, 5, -1) # [[[h, e, l, l, o]]] 그대로 사용 
out, hidden = cell(inputs, hidden)
print("sequence input size", inputs.size(), "out size", out.size())
# sequence input size torch.Size([1, 5, 4]) out size torch.Size([1, 5, 2])

#-----------------------------------batch size 3--------------------------------------#

# hidden : (num_layers * num_directions, batch, hidden_size) whether batch_first=True or False
hidden = Variable(torch.randn(1, 3, 2))

# One cell RNN input_dim (4) -> output_dim (2). sequence: 5, batch 3
# 3 batches 'hello', 'eolll', 'lleel'
# rank = (3, 5, 4) -> (batch_size, sequence_length, input_size)  // ex) h -- [1,0,0,0]
inputs = Variable(torch.Tensor([[h, e, l, l, o],
                                [e, o, l, l, l],
                                [l, l, e, e, l]])) # (3, 5, 4)

# Propagate input through RNN
# Input: (batch, seq_len, input_size) when batch_first=True
# B x S x I
out, hidden = cell(inputs, hidden)
print("batch input size", inputs.size(), "out size", out.size())
# batch input size torch.Size([3, 5, 4]) out size torch.Size([3, 5, 2])

# One cell RNN input_dim (4) -> output_dim (2)
cell = nn.RNN(input_size=4, hidden_size=2)

# The given dimensions dim0 and dim1 are swapped.
inputs = inputs.transpose(dim0=0, dim1=1)
# Propagate input through RNN
# Input: (seq_len, batch_size, input_size) when batch_first=False (default)
# S x B x I
out, hidden = cell(inputs, hidden)
print("batch input size", inputs.size(), "out size", out.size())

# batch size <-> Sequence num ----------------------------------- 무슨 의미인지 잘 모르겠다.
# batch input size torch.Size([3, 5, 4]) -> ([5, 3, 4])
# out size torch.Size        ([3, 5, 2]) -> ([5, 3, 2])

