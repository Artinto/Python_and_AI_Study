import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from name_dataset import NameDataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Parameters and DataLoaders
HIDDEN_SIZE = 100 # 히든 사이즈는 100
N_CHARS = 128 # ASCII
N_CLASSES = 18 # 18개국


class RNNClassifier(nn.Module): # RNN 분류기(다수의 입력, 출력이 1개)

    def __init__(self, input_size, hidden_size, output_size, n_layers=1): # 초기화 메서드( )
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, hidden_size) # 임베딩 입력, 출력 대입
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers) # GRU 사용 입력, 출력이 같음
        self.fc = nn.Linear(hidden_size, output_size) # Fully Connected Layer, 입력, 출력

    def forward(self, input): # input을 받아서
        # Note: we run this all at once (over the whole input sequence)

        # input = B x S . size(0) = B
        batch_size = input.size(0) # batch 사이즈는 input의 첫번째 차원

        # input:  B x S  -- (transpose) --> S x B
        input = input.t() # GRU에서는 1차원 형식의 리스트 행렬을 사용하지 않으므로, 전치 행렬을 통해 행렬변경

        # Embedding S x B -> S x B x I (embedding size)
        print("  input", input.size()) # 임베딩 후의 텐서 사이즈 출력 S x B
        embedded = self.embedding(input) # 임베딩을 생성
        print("  embedding", embedded.size()) # 임베딩 후의 텐서 사이즈 출력 S x B x I

        # Make a hidden
        hidden = self._init_hidden(batch_size)

        output, hidden = self.gru(embedded, hidden) # GRU: embedded입력 output출력, hidden 입력및 출력
        print("  gru hidden output", hidden.size())
        # Use the last layer output as FC's input
        # No need to unpack, since we are going to use hidden
        fc_output = self.fc(hidden) # fully connected를 통해 마지막 output값만을 사용, (마지막 rnn부분 hidden=output)
        print("  fc output", fc_output.size())
        return fc_output

    def _init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size) # 초기 hidden을 0행렬로 만들어줌
        return Variable(hidden)

# Help functions


def str2ascii_arr(msg): # 주어진 이름들을 아스키 코드 리스트로 바꿔줌
    arr = [ord(c) for c in msg] # ord를 통해 문자를 아스키코드로 변경
    return arr, len(arr)

# pad sequences and sort the tensor
def pad_sequences(vectorized_seqs, seq_lengths): # zero padding
    seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()
    for idx, (seq, seq_len) in enumerate(zip(vectorized_seqs, seq_lengths)):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)
    return seq_tensor

# Create necessary variables, lengths, and target
def make_variables(names): # 이름을 변수로 바꾸는 함수
    sequence_and_length = [str2ascii_arr(name) for name in names] # 각 글자 아스키코드의 배열들을 리스트로 묶음
    vectorized_seqs = [sl[0] for sl in sequence_and_length] # 각 이름 아스키코드들의 배열
    seq_lengths = torch.LongTensor([sl[1] for sl in sequence_and_length]) # 각 이름 아스키코드 크기의 배열
    return pad_sequences(vectorized_seqs, seq_lengths) # 아스키코드 배열 및 그 크기들의 배열을 pad_sequences에 넣은 값을 리턴


if __name__ == '__main__':
    names = ['adylov', 'solan', 'hard', 'san'] # 한번에 여러 이름을 처리하기 위하여 names리스트에 넣어줌
    classifier = RNNClassifier(N_CHARS, HIDDEN_SIZE, N_CLASSES)

    for name in names:
        arr, _ = str2ascii_arr(name) # arr을 arr에 , len(arr)를 _에 넣는다??
        inp = Variable(torch.LongTensor([arr])) # arr을 텐서로 만들어줌
        out = classifier(inp) 
        print("in", inp.size(), "out", out.size())


    inputs = make_variables(names) # make_variables함수를 통해 input에 아스키코드 값 대입
    out = classifier(inputs) # out은 classifier에 input을 넣은 것
    print("batch in", inputs.size(), "batch out", out.size())
