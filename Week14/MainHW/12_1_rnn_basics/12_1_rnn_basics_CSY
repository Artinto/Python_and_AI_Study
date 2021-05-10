# torch 모듈 불러옴
# touch 모듈 내에 있는 nn을 불러옴
# Tensor에 정의된 거의 모든 연산 지원
import torch
import torch.nn as nn
from torch.autograd import Variable

# One hot encoding for each char in 'hello'
# 각 알파벳을 원 핫 인코딩
h = [1, 0, 0, 0]
e = [0, 1, 0, 0]
l = [0, 0, 1, 0]
o = [0, 0, 0, 1]

# One cell RNN input_dim (4) -> output_dim (2). sequence: 5
# input 벡터 크기: 4, output 벡터 크기: 2, 모델에 데이터를 넣을 때 배치 사이즈가 가장 먼저 나오도록 함
cell = nn.RNN(input_size=4, hidden_size=2, batch_first=True)

# (num_layers * num_directions, batch, hidden_size) whether batch_first=True or False
#  1 layer과 1 direction이기 때문에 1, 배치사이즈 1, output size 2 / torch.randn()를 사용하여 랜덤 데이터와 제공된 차원으로 텐서 생성
hidden = Variable(torch.randn(1, 1, 2))

# Propagate input through RNN
# Input: (batch, seq_len, input_size) when batch_first=True
# 입력으로 텐서 h, e, l, l ,o를 줌
# 입력의 배치 사이즈 1, 길이 1로 설정/ input size로 -1은 해당 값 유추하여 넣음
inputs = Variable(torch.Tensor([h, e, l, l, o]))
for one in inputs:
    one = one.view(1, 1, -1)
    # Input: (batch, seq_len, input_size) when batch_first=True
    # out변수와 hidden변수에 cell 연산한 결과 저장(hidden은 다음 cell넘겨줄 hidden)
    out, hidden = cell(one, hidden)
    print("one input size", one.size(), "out size", out.size())

# We can do the whole at once
# Propagate input through RNN
# Input: (batch, seq_len, input_size) when batch_first=True
# 입력의 배치 사이즈 1, 길이 5로 설정/ input size로 -1은 해당 값 유추하여 넣음
# out변수와 hidden변수에 cell 연산한 결과 저장(hidden은 다음 cell넘겨줄 hidden)
inputs = inputs.view(1, 5, -1)
out, hidden = cell(inputs, hidden)
print("sequence input size", inputs.size(), "out size", out.size())


# hidden : (num_layers * num_directions, batch, hidden_size) whether batch_first=True or False
#  1 layer과 1 direction이기 때문에 1, 배치사이즈 3, output size 2 / torch.randn()를 사용하여 랜덤 데이터와 제공된 차원으로 텐서 생성
hidden = Variable(torch.randn(1, 3, 2))

# One cell RNN input_dim (4) -> output_dim (2). sequence: 5, batch 3
# 3 batches 'hello', 'eolll', 'lleel'
# rank = (3, 5, 4)
# 입력 변수 inputs에 텐서 저장 (3개의 배치, 5개의 sequence)
inputs = Variable(torch.Tensor([[h, e, l, l, o],
                                [e, o, l, l, l],
                                [l, l, e, e, l]]))

# Propagate input through RNN
# Input: (batch, seq_len, input_size) when batch_first=True
# B x S x I
# out변수와 hidden변수에 cell 연산한 결과 저장(hidden은 다음 cell넘겨줄 hidden)
out, hidden = cell(inputs, hidden)
print("batch input size", inputs.size(), "out size", out.size())


# One cell RNN input_dim (4) -> output_dim (2)
# 입력 사이즈는 4, 출력 사이즈는 2로 cell에 저장
cell = nn.RNN(input_size=4, hidden_size=2)

# The given dimensions dim0 and dim1 are swapped.
# dim0의 차원과 dim1의 차원을 교환
inputs = inputs.transpose(dim0=0, dim1=1)
# Propagate input through RNN
# Input: (seq_len, batch_size, input_size) when batch_first=False (default)
# S x B x I
# out변수와 hidden변수에 cell 연산한 결과 저장(hidden은 다음 cell넘겨줄 hidden)
out, hidden = cell(inputs, hidden)
print("batch input size", inputs.size(), "out size", out.size())
