# Original code is from https://github.com/spro/practical-pytorch
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence 
from name_dataset import NameDataset


# Parameters and DataLoaders
HIDDEN_SIZE = 100 
N_CHARS = 128
N_CLASSES = 18 


class RNNClassifier(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, hidden_size) #벡터로 변환
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers) #GRU함수 사용 hiddenzise = outpustsize
        self.fc = nn.Linear(hidden_size, output_size) #Fully Connected

    def forward(self, input):
        # Note: we run this all at once (over the whole input sequence)

        # input = B x S . size(0) = B
        batch_size = input.size(0)

        # input:  B x S  -- (transpose) --> S x B
        input = input.t() #트렌스포즈

        # Embedding S x B -> S x B x I (embedding size)
        print("  input", input.size())
        embedded = self.embedding(input)
        print("  embedding", embedded.size()) # S*B*I

        # Make a hidden
        hidden = self._init_hidden(batch_size) #초기화

        output, hidden = self.gru(embedded, hidden) #embedded 에 s*b*i들어감. GRU함수 사용
        print("  gru hidden output", hidden.size())
        # Use the last layer output as FC's input
        # No need to unpack, since we are going to use hidden
        fc_output = self.fc(hidden) #fc로 나열
        print("  fc output", fc_output.size())
        return fc_output

    def _init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        return Variable(hidden)

# Help functions


def str2ascii_arr(msg): #아스키로 변환 함수
    arr = [ord(c) for c in msg]
    return arr, len(arr)

# pad sequences and sort the tensor
def pad_sequences(vectorized_seqs, seq_lengths): #0 패딩으로 사이즈 맞춰주기
    seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()
    for idx, (seq, seq_len) in enumerate(zip(vectorized_seqs, seq_lengths)):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)
    return seq_tensor

# Create necessary variables, lengths, and target
def make_variables(names): 
    sequence_and_length = [str2ascii_arr(name) for name in names]
    vectorized_seqs = [sl[0] for sl in sequence_and_length] 
    seq_lengths = torch.LongTensor([sl[1] for sl in sequence_and_length]) 
    return pad_sequences(vectorized_seqs, seq_lengths)


if __name__ == '__main__':
    names = ['adylov', 'solan', 'hard', 'san']
    classifier = RNNClassifier(N_CHARS, HIDDEN_SIZE, N_CLASSES)

    for name in names:
        arr, _ = str2ascii_arr(name) #입력
        inp = Variable(torch.LongTensor([arr])) #텐서로변경
        out = classifier(inp) 
        print("in", inp.size(), "out", out.size())


    inputs = make_variables(names) #names들을 batch로 집어넣음
    ut = classifier(inputs)
    print("batch in", inputs.size(), "batch out", out.size())
