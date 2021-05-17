import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

#from name_dataset import NameDataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

HIDDEN_SIZE=100 #
N_CHARS=128 #아스키 코드의 개수
N_CLASSES=18 #18개국

class RNNClassifier(nn.Module):
  def __init__(self, input_size, hidden_size, output_size, n_layers=1):#초기화
    super(RNNClassifier, self).__init__()
    self.hidden_size=hidden_size
    self.n_layers=n_layers

    self.embedding=nn.Embedding(input_size, hidden_size)#알파벳을 아스키코드로 바꾸어서 밀집 벡터를 만든다. 단어의 크기(128)와 rnn으로 들어갈 벡터의 차원을 적는다 input*hidden 행렬 생성
    self.gru=nn.GRU(hidden_size, hidden_size, n_layers)# rnn cell을 거치면서 생기는 문제를 해결하기 위해 나온 cell,  input*hidden*1 행렬 생성
    self.fc=nn.Linear(hidden_size, output_size)#선형회귀

  def forward(self, input):
    batch_size=input.size(0)#input은 batchsize*S
    input=input.t() #행렬변환 
    print("input", input.size())#input의 변환 후 크기
    embedded=self.embedding(input)#임베딩
    print("embedding", embedded.size())#임베딩 후 텐서 사이즈 S*batchsize*
    hidden=self._init_hidden(batch_size)#hidden 행렬 생성
    output, hidden=self.gru(embedded, hidden)#input과 행렬 넣고 rnn
    print("gru hidden output", hidden.size())
    fc_output=self.fc(hidden)#linnear에 넣기 outputsize에 맞추기 위해
    print("fc output", fc_output.size())

    return fc_output

  def _init_hidden(self, batch_size):
    hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size)#행렬생성
    return Variable(hidden)

def str2ascii_arr(msg):
  arr=[ord(c) for c in msg]#문자가 아스키 코드로 바뀜
  return arr, len(arr)

def pad_sequences(vectorized_seqs, seq_lengths):
  seq_tensor=torch.zeros((len(vectorized_seqs), seq_lengths.max())).long() #행렬을 만들고 long타입으로, 크기가 다 다를 수 있으므로 zero로
  for idx, (seq, seq_len) in enumerate(zip(vectorized_seqs, seq_lengths)): #몇번째 반복 문인지 확인
    seq_tensor[idx, :seq_len]=torch.LongTensor(seq)#텐서를 만듦 
  return seq_tensor

def make_variables(names):
  sequence_and_length=[str2ascii_arr(name) for name in names]#여러 문자들은 아스키코드로 만듦
  vectorized_seqs=[sl[0] for sl in sequence_and_length]#0번째에 하나씩 저장
  seq_lengths=torch.LongTensor([sl[1] for sl in sequence_and_length])#길이를 longtensor로 저장
  return pad_sequences(vectorized_seqs, seq_lengths)

names=['adylov', 'solan', 'hard', 'san']#name list를 만들어줌
classifier=RNNClassifier(N_CHARS, HIDDEN_SIZE, N_CLASSES)

for name in names:
  arr, _ =str2ascii_arr(name)#arr만 필요
  inp=Variable(torch.LongTensor([arr]))#tensor로 바꿈
  out=classifier(inp)#아스키코드로 바꿈
  print("in", inp.size(), "out", out.size())

  inputs=make_variables(names)
  out=classifier(inputs)

  print("batch in", inputs.size(), "batch out", out.size())
