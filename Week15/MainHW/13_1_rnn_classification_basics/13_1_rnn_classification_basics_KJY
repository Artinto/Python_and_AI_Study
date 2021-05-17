# Original code is from https://github.com/spro/practical-pytorch
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from name_dataset import NameDataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Parameters and DataLoaders
HIDDEN_SIZE = 100 # alpha size/hidden size : 출력값의 vector 크기는 100이라는 뜻

N_CHARS = 128  # ASCII
N_CLASSES = 18 # 18개 나라이름


class RNNClassifier(nn.Module): # 다수 입력, 하나의 출력 (RNN 타입 중 하나)

    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(RNNClassifier, self).__init__() 
        self.hidden_size = hidden_size 
        self.n_layers = n_layers # n_layers 가 1인 이유..?

        self.embedding = nn.Embedding(input_size, hidden_size) # Embedding 통해 0과 1 사이의 소수로 표현
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers) # hidden_size : RNN 의 input size 이자 output size
        self.fc = nn.Linear(hidden_size, output_size) # fully connected layer - hidden_size :input (128) , output_size : output (18)

    def forward(self, input): # 모든 행위를 연결
        # Note: we run this all at once (over the whole input sequence)

        # input = B x S (Batch size x sequence). size(0) = B
        batch_size = input.size(0) 

        # input:  B x S  -- (transpose) --> S x B 
        input = input.t() # 행과 열 교환

        # Embedding S x B -> S x B x I(I :input size) (embedding size)
        print("  input", input.size())
        embedded = self.embedding(input) # 0과 1 사이의 소수로 바꿈 -> gru 의 입력값
        print("  embedding", embedded.size())

        # Make a hidden
        hidden = self._init_hidden(batch_size) # _init_hidden : hidden 의 초기값 (0으로만 구성)

        output, hidden = self.gru(embedded, hidden) # 마지막 output 만 사용가능 
        print("  gru hidden output", hidden.size())
        # Use the last layer output as FC's input
        # No need to unpack, since we are going to use hidden
        fc_output = self.fc(hidden) # fully connected layer 통과한 last output
        print("  fc output", fc_output.size())
        return fc_output

    def _init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size) # hidden 초기값 : 행렬을 다 0으로 만듦
        return torch.FloatTensor(hidden) # Tensor 로 바꿔줌

# Help functions


def str2ascii_arr(msg):
    arr = [ord(c) for c in msg] # ord : ascii로 변환 후 리스트로 만듦
    return arr, len(arr) # 배열과 길이를 리턴

# pad sequences and sort the tensor
# zero padding
def pad_sequences(vectorized_seqs, seq_lengths):
    seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long() # 실수 표현 범위 ->64bit 정수, n*m 행렬로 만드는 작업(빈 공간 0으로 채워줌)
    for idx, (seq, seq_len) in enumerate(zip(vectorized_seqs, seq_lengths)): # 동일한 개수로 구성된 자료를 하나로 묶어주는 함수
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)
    return seq_tensor

# Create necessary variables, lengths, and target
# 이름을 변수로 바꿔주는 함수
def make_variables(names): # names : 이름에 대한 배열 받는 파라미터
    sequence_and_length = [str2ascii_arr(name) for name in names] # ascii 값으로 변환한 것과 길이가 포함된 리스트 생성
    vectorized_seqs = [sl[0] for sl in sequence_and_length] # ascii 리스트 저장
    seq_lengths = torch.LongTensor([sl[1] for sl in sequence_and_length]) # 길이에 대한 리스트 저장
    return pad_sequences(vectorized_seqs, seq_lengths)


if __name__ == '__main__':
    names = ['adylov', 'solan', 'hard', 'san']
    classifier = RNNClassifier(N_CHARS, HIDDEN_SIZE, N_CLASSES)

    for name in names:
        arr, _ = str2ascii_arr(name) # _ :length of array다??
        inp = torch.FloatTensor(torch.LongTensor([arr]))
        out = classifier(inp)
        print("in", inp.size(), "out", out.size())


    inputs = make_variables(names)
    out = classifier(inputs)
    print("batch in", inputs.size(), "batch out", out.size())
