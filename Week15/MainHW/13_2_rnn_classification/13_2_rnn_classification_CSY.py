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
# N_LAYERS, BATCH_SIZE, N_EPOCHS 설정
HIDDEN_SIZE = 100
N_LAYERS = 2
BATCH_SIZE = 256
N_EPOCHS = 100

# train_set=False로 하여 test set으로 전환
# test data를 로드, 무작위 
test_dataset = NameDataset(is_train_set=False)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE, shuffle=True)

# train_set=True로 하여 train set으로 전환
# train data를 로드, 무작위 
train_dataset = NameDataset(is_train_set=True)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE, shuffle=True)

# 국가 개수(18)를 변수에 저장
# 아스키 범위가 0부터 127까지 이기 때문에 N_CHARS는 128
N_COUNTRIES = len(train_dataset.get_countries())
print(N_COUNTRIES, "countries")
N_CHARS = 128  # ASCII


# Some utility functions
# 걸린 시간 측정 함수
def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# variable 생성 함수
def create_variable(tensor):
    # Do cuda() before wrapping with variable
    # cuda를 사용한다면 tersor.cuda()로, 아니라면 tensor로
    if torch.cuda.is_available():
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)


# pad sequences and sort the tensor
# sequence들의 길이가 달라 zero padding을 이용하여 해결
def pad_sequences(vectorized_seqs, seq_lengths, countries):
    # sequence 길이가 제일 긴 sequece의 길이를 기준으로 for문을 이용하여 zero padding을 함
    seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()
    for idx, (seq, seq_len) in enumerate(zip(vectorized_seqs, seq_lengths)):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)

    # Sort tensors by their length
    # sequence 길이를 내림차순으로 정렬
    # 제일 긴 sequence를 seq_tensor에 저장
    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    seq_tensor = seq_tensor[perm_idx]

    # Also sort the target (countries) in the same order
    # 같은 방법으로 국가 정렬
    target = countries2tensor(countries)
    if len(countries):
        target = target[perm_idx]

    # Return variables
    # DataParallel requires everything to be a Variable
    # 각 변수들 반환
    return create_variable(seq_tensor), \
        create_variable(seq_lengths), \
        create_variable(target)


# Create necessary variables, lengths, and target
# 필요한 변수와 길이들, target을 사용자가 생성함
def make_variables(names, countries):
    sequence_and_length = [str2ascii_arr(name) for name in names]
    vectorized_seqs = [sl[0] for sl in sequence_and_length]
    seq_lengths = torch.LongTensor([sl[1] for sl in sequence_and_length])
    return pad_sequences(vectorized_seqs, seq_lengths, countries)

#아스키 코드로 변환 함수
#msg에서 char를 하나씩 꺼내와 ord 함수를 통해 아스키 코드로 변환 후 arr배열에 저장
# 배열과 배열 길이 반환
def str2ascii_arr(msg):
    arr = [ord(c) for c in msg]
    return arr, len(arr)

# 길이가 긴 country tensor를 for문을 이용하여 찾기
def countries2tensor(countries):
    country_ids = [train_dataset.get_country_id(
        country) for country in countries]
    return torch.LongTensor(country_ids)

# torch.nn.Module을 상속받는 파이썬 클래스 정의
# 모델의 생성자 정의 (객체가 갖는 속성값을 초기화하는 역할, 객체가 생성될 때 자동으로 호출됨)/bidirectional=True ->BiLSTM 사용 가능
class RNNClassifier(nn.Module):
    # Our model

    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=True):
        # super() 함수를 통해 nn.Module의 생성자를 호출함
        # hidden size 초기화 100
        # n_layers 초기화 2
        # n_directions 초기화
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = int(bidirectional) + 1

        # 임베딩 레이어 선언 input 사이즈와 output 사이즈가 인자
        # 순환 신경망 GRU에 input 사이즈와 output 사이즈, 레이어 수가 인자를 넘김
        # 입력은 hidden 사이즈, 출력은 output 사이즈(18)로 신경망 계층 생성(fully-connected layer)
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size, output_size)

 # 모델 객체와 input, seq_lengths를 받아 forward 연산하는 함수로 객체를 호출하면 자동으로 forward 연산 수행됨
    def forward(self, input, seq_lengths):
        # Note: we run this all at once (over the whole input sequence)
        # input shape: B x S (input size)
        # transpose to make S(sequence) x B (batch)
        # input을 transpose
        # input 사이즈는 sequence x batch /size()로 배치 사이즈 설정
        input = input.t()
        batch_size = input.size(1)

        # hidden 생성
        # Make a hidden
        hidden = self._init_hidden(batch_size)

        # Embedding S x B -> S x B x I (embedding size)
        # 임베딩한 결과를 embedded에 저장/embedding을 하면 사이즈가 sequence x batch에서 sequence x batch x input으로 변환
        embedded = self.embedding(input)

        # Pack them up nicely
        # packed sequence 과정(매 배치마다 고정된 길이로 만들기 위해 정렬 후 통합된 배치로 만듦)
        gru_input = pack_padded_sequence(
            embedded, seq_lengths.data.cpu().numpy())

        # To compact weights again call flatten_parameters().
        # flatten_parameters()를 이용해 가중치 복사
        # 순환 신경망 gru를 거친 결과값을 각각 output, hidden에 저장
        self.gru.flatten_parameters()
        output, hidden = self.gru(gru_input, hidden)

        # Use the last layer output as FC's input
        # No need to unpack, since we are going to use hidden
        # 마지막 레이어의 output을 반환
        fc_output = self.fc(hidden[-1])
        return fc_output

    # hidden 초기화
    def _init_hidden(self, batch_size):
        # 모델 처음 시작시 hidden zero 배열로 초기화
        hidden = torch.zeros(self.n_layers * self.n_directions,
                             batch_size, self.hidden_size)
        return create_variable(hidden)


# Train cycle
# train 함수
def train():
    # loss 값 초기화
    total_loss = 0

    # input와 target 변수 설정
    # input, seq_lengths를 인자로 넘겨 예측값 output을 도출
    for i, (names, countries) in enumerate(train_loader, 1):
        input, seq_lengths, target = make_variables(names, countries)
        output = classifier(input, seq_lengths)

        # output과 target의 loss값을 구함
        # loss값을 다 합산
        loss = criterion(output, target)
        total_loss += loss.data[0]

        # 미분을 통해 얻은 기울기가 이전에 계산된 기울기 값에 누적되기 때문에 기울기 값을 0으로 초기화 함
        # 역전파 실행, 손실 함수를 미분하여 기울기 계산
        # step() 함수를 호출하여 parameter를 업데이트함
        classifier.zero_grad()
        loss.backward()
        optimizer.step()

        # 배치 사이즈가 10의 배수일때마다 loss값 출력
        if i % 10 == 0:
            print('[{}] Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.2f}'.format(
                time_since(start), epoch,  i *
                len(names), len(train_loader.dataset),
                100. * i * len(names) / len(train_loader.dataset),
                total_loss / i * len(names)))

    # 전체 loss값 반환
    return total_loss


# Testing cycle
# 테스트
def test(name=None):
    # Predict for a given name
    if name:
        # make_variables 함수를 통해 입력, sequence 길이, target을 생성
        # 학습한 모델을 이용하여 test 실행 / output을 얻음
        # 최대값의 index를 반환/keepdim은 벡터 차원을 유지 시킬건지 아닌지를 설정
        # country_id에 예측 값 저장
        # 예측한 이름 출력
        input, seq_lengths, target = make_variables([name], [])
        output = classifier(input, seq_lengths)
        pred = output.data.max(1, keepdim=True)[1]
        country_id = pred.cpu().numpy()[0][0]
        print(name, "is", train_dataset.get_country(country_id))
        return

    # 정확도 계산을 위한 변수와 train data size 선언
    print("evaluating trained model ...")
    correct = 0
    train_data_size = len(test_loader.dataset)

    # test 데이터를 for문을 이용하여 하나씩 넘김
    for names, countries in test_loader:
        # make_variables 함수를 통해 입력, sequence 길이, target을 생성
        # 학습한 모델을 이용하여 test 실행 / output을 얻음
        # 최대값의 index를 반환/keepdim은 벡터 차원을 유지 시킬건지 아닌지를 설정
        # 맞은 개수를 더함
        input, seq_lengths, target = make_variables(names, countries)
        output = classifier(input, seq_lengths)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    # 정확도 출력
    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, train_data_size, 100. * correct / train_data_size))

# main함수
if __name__ == '__main__':

    # classifier 클래스 객체 선언 (인스턴스화)
    classifier = RNNClassifier(N_CHARS, HIDDEN_SIZE, N_COUNTRIES, N_LAYERS)
    # cuda를 사용한다면 출력문 실행
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [33, xxx] -> [11, ...], [11, ...], [11, ...] on 3 GPUs
        # DataParallel()를 사용하여 모델 래핑(GPU를 이용하도록 함)
        classifier = nn.DataParallel(classifier)

    # cuda를 사용한다면 실행
    if torch.cuda.is_available():
        classifier.cuda()

    # 최적화로 아담 사용, 학습률 0.001
    # 손실함수는 Cross Entropy
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 시작 시간 측정
    start = time.time()
    print("Training for %d epochs..." % N_EPOCHS)
    for epoch in range(1, N_EPOCHS + 1):
        # Train cycle
        # 학습 실행
        train()

        # Testing
        # 테스트 실행
        test()

        # Testing several samples
        # 테스트 예시
        test("Sung")
        test("Jungwoo")
        test("Soojin")
        test("Nako")
