# Original code is from https://github.com/spro/practical-pytorch
# time 모듈 불러옴-시간 데이터를 다룸
# math 모듈 불러옴-수학 관련 함수
# torch 모듈 불러옴
# touch 모듈 내에 있는 nn을 불러옴
# 미분 값을 자동 계산/자동 계산을 위한 변수 Variable
# 데이터를 불러오기 위한 클래스
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

# Name Dataset을 설정하기 위한 모듈 선언
# zero padding을 위한 모듈 선언
from name_dataset import NameDataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Parameters and DataLoaders
# hidden 사이즈 100
# 아스키 범위가 0부터 127까지 이기 때문에 N_CHARS는 128
# 18개의 국가이기 때문에 N_CLASSES는 18
HIDDEN_SIZE = 100
N_CHARS = 128  # ASCII
N_CLASSES = 18

# torch.nn.Module을 상속받는 파이썬 클래스 정의
# 모델의 생성자 정의 (객체가 갖는 속성값을 초기화하는 역할, 객체가 생성될 때 자동으로 호출됨)
class RNNClassifier(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        # super() 함수를 통해 nn.Module의 생성자를 호출함
        # hidden size 초기화 100
        # n_layers 초기화 1
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        # 임베딩 레이어 선언 input 사이즈와 output 사이즈가 인자
        # 순환 신경망 GRU에 input 사이즈와 output 사이즈, 레이어 수가 인자를 넘김
        # 입력은 hidden 사이즈, 출력은 output 사이즈로 신경망 계층 생성(fully-connected layer)
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        self.fc = nn.Linear(hidden_size, output_size)

 # 모델 객체와 input을 받아 forward 연산하는 함수로 객체를 호출하면 자동으로 forward 연산 수행됨
    def forward(self, input):
        # Note: we run this all at once (over the whole input sequence)

        # input = B x S . size(0) = B
        # input 사이즈는 batch x sequence / size()로 배치 사이즈 설정
        batch_size = input.size(0)

        # input:  B x S  -- (transpose) --> S x B
        # input을 transpose
        input = input.t()

        # Embedding S x B -> S x B x I (embedding size)
        print("  input", input.size())
        # 임베딩한 결과를 embedded에 저장/embedding을 하면 사이즈가 sequence x batch에서 sequence x batch x input으로 변환
        embedded = self.embedding(input)
        print("  embedding", embedded.size())

        # Make a hidden
        # hidden 생성
        hidden = self._init_hidden(batch_size)

        # 순환 신경망 gru를 거친 결과값을 각각 output, hidden에 저장
        output, hidden = self.gru(embedded, hidden)
        print("  gru hidden output", hidden.size())
        # Use the last layer output as FC's input
        # No need to unpack, since we are going to use hidden
        # 마지막 레이어의 output이 fully connencted layer에 input이 됨
        fc_output = self.fc(hidden)
        print("  fc output", fc_output.size())
        return fc_output

    # hidden 초기화
    def _init_hidden(self, batch_size):
        # 모델 처음 시작시 hidden zero 배열로 초기화
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        return Variable(hidden)

# Help functions

#아스키 코드로 변환 함수
#msg에서 char를 하나씩 꺼내와 ord 함수를 통해 아스키 코드로 변환 후 arr배열에 저장
# 배열과 배열 길이 반환
def str2ascii_arr(msg):
    arr = [ord(c) for c in msg]
    return arr, len(arr)

# pad sequences and sort the tensor
# sequence들의 길이가 달라 zero padding을 이용하여 해결
def pad_sequences(vectorized_seqs, seq_lengths):
    # sequence 길이가 제일 긴 sequece의 길이를 기준으로 for문을 이용하여 zero padding을 함
    seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()
    for idx, (seq, seq_len) in enumerate(zip(vectorized_seqs, seq_lengths)):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)
    return seq_tensor

# Create necessary variables, lengths, and target
# 필요한 변수와 길이들, target을 사용자가 생성함
def make_variables(names):
    sequence_and_length = [str2ascii_arr(name) for name in names]
    vectorized_seqs = [sl[0] for sl in sequence_and_length]
    seq_lengths = torch.LongTensor([sl[1] for sl in sequence_and_length])
    return pad_sequences(vectorized_seqs, seq_lengths)

# main함수
if __name__ == '__main__':
    # 이름들을 리스트로 저장
    # classifier 클래스 객체 선언 (인스턴스화)
    names = ['adylov', 'solan', 'hard', 'san']
    classifier = RNNClassifier(N_CHARS, HIDDEN_SIZE, N_CLASSES)

    # for문으로 names에 있는 이름을 하나씩 꺼내옴
    for name in names:
        # 아스키 코드로 변환하여 배열에 저장
        # inp에 배열을 텐서로 변환하여 저장
        # 모델을 거친 결과를 out에 저장
        arr, _ = str2ascii_arr(name)
        inp = Variable(torch.LongTensor([arr]))
        out = classifier(inp)
        print("in", inp.size(), "out", out.size())

    # make_variables에 names리스트로 인자로 넘긴 후 그 결과값을 inputs에 저장
    # 모델을 거친 결과를 out에 저장
    inputs = make_variables(names)
    out = classifier(inputs)
    print("batch in", inputs.size(), "batch out", out.size())
