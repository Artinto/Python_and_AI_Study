# Original code is from https://github.com/spro/practical-pytorch
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from name_dataset import NameDataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence #packing과 unpacking

# Parameters and DataLoaders
HIDDEN_SIZE = 100 
N_CHARS = 128  #아스키코드
N_CLASSES = 18 #나라갯수


class RNNClassifier(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers #1계층

        self.embedding = nn.Embedding(input_size, hidden_size) #임베딩, 문자에 해당된 정수를 밀집벡터로 변환
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers) #GRU사용
        self.fc = nn.Linear(hidden_size, output_size) #18개로 분류

    def forward(self, input):
        # Note: we run this all at once (over the whole input sequence)

        # input = B x S . size(0) = B
        batch_size = input.size(0) #배치 사이즈 설정

        # input:  B x S  -- (transpose) --> S x B
        input = input.t() #행과 열 교환

        # Embedding S x B -> S x B x I (embedding size)
        print("  input", input.size())
        embedded = self.embedding(input) #임베딩
        print("  embedding", embedded.size())

        # Make a hidden
        hidden = self._init_hidden(batch_size) #첫 히든값 초기화

        output, hidden = self.gru(embedded, hidden) #임베딩된 벡터와 히든값으로 oputput과 다음 히든값 도출
        print("  gru hidden output", hidden.size())
        # Use the last layer output as FC's input
        # No need to unpack, since we are going to use hidden
        fc_output = self.fc(hidden) #마지막 히든과 output은 동일함
        print("  fc output", fc_output.size())
        return fc_output

    def _init_hidden(self, batch_size): #zero vector로 초기화
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        return Variable(hidden)

# Help functions


def str2ascii_arr(msg): #문자열을 아스키 정수로 변환
    arr = [ord(c) for c in msg]
    return arr, len(arr)

# pad sequences and sort the tensor
def pad_sequences(vectorized_seqs, seq_lengths): #각각의 문자열의 크기가 다르기 때문에 제일 긴 문자열을 기준으로 0 padding
    seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()
    for idx, (seq, seq_len) in enumerate(zip(vectorized_seqs, seq_lengths)):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)
    return seq_tensor

# Create necessary variables, lengths, and target
def make_variables(names): 
    sequence_and_length = [str2ascii_arr(name) for name in names] #아스키코드 정수 변환
    vectorized_seqs = [sl[0] for sl in sequence_and_length] 
    seq_lengths = torch.LongTensor([sl[1] for sl in sequence_and_length]) 
    return pad_sequences(vectorized_seqs, seq_lengths)


if __name__ == '__main__':
    names = ['adylov', 'solan', 'hard', 'san']
    classifier = RNNClassifier(N_CHARS, HIDDEN_SIZE, N_CLASSES)

    for name in names:
        arr, _ = str2ascii_arr(name) #아스키코드로 변환
        inp = Variable(torch.LongTensor([arr])) #정수이기때문에 longtensor사용
        out = classifier(inp) 
        print("in", inp.size(), "out", out.size())


    inputs = make_variables(names)
    out = classifier(inputs)
    print("batch in", inputs.size(), "batch out", out.size())
