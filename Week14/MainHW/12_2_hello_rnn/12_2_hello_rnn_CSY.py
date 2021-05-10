# Lab 12 RNN
# 파이썬 인터프리터 제어 모듈 불러옴
# torch 모듈 불러옴
# touch 모듈 내에 있는 nn을 불러옴
# Tensor에 정의된 거의 모든 연산 지원
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable

torch.manual_seed(777)  # reproducibility
#            0    1    2    3    4
# 각 알파벳에 인덱스 부여
idx2char = ['h', 'i', 'e', 'l', 'o']

# Teach hihell -> ihello
# hihell을 x_data로 설정하여 ihello를 예측할 수 있도록 함
# 원 핫 인코딩 설정
# for loop을 이용하여 원 핫 인코딩한 x_one_hot 생성
x_data = [0, 1, 0, 2, 3, 3]   # hihell
one_hot_lookup = [[1, 0, 0, 0, 0],  # 0
                  [0, 1, 0, 0, 0],  # 1
                  [0, 0, 1, 0, 0],  # 2
                  [0, 0, 0, 1, 0],  # 3
                  [0, 0, 0, 0, 1]]  # 4

# 결과값인 ihello를 y_data에 설정
y_data = [1, 0, 2, 3, 3, 4]    # ihello
x_one_hot = [one_hot_lookup[x] for x in x_data]

# As we have one batch of samples, we will change them to variables only once
# 입력(inputs)을 텐서로 만들어 저장
# 결과(labels)를 텐서로 만들어 저장
inputs = Variable(torch.Tensor(x_one_hot))
labels = Variable(torch.LongTensor(y_data))

# 클래스 개수 5개, input 크기 5, output 크기 5, 배치 사이즈 1, sequence 길이 1, layer 개수 1 ->변수로 저장
num_classes = 5
input_size = 5  # one-hot size
hidden_size = 5  # output from the RNN. 5 to directly predict one-hot
batch_size = 1   # one sentence
sequence_length = 1  # One by one
num_layers = 1  # one-layer rnn

# torch.nn.Module을 상속받는 파이썬 클래스 정의
# 모델의 생성자 정의 (객체가 갖는 속성값을 초기화하는 역할, 객체가 생성될 때 자동으로 호출됨)
class Model(nn.Module):

    def __init__(self):
        # super() 함수를 통해 nn.Module의 생성자를 호출함
        # rnn 초기화
        super(Model, self).__init__()
        self.rnn = nn.RNN(input_size=input_size,
                          hidden_size=hidden_size, batch_first=True)

    # 모델 객체와 x, hidden state를 입력으로 받음
    def forward(self, hidden, x):
        # Reshape input (batch first)
        # 입력 shape를 재설정 (배치 사이즈가 처음으로)
        x = x.view(batch_size, sequence_length, input_size)

        # Propagate input through RNN
        # Input: (batch, seq_len, input_size)
        # hidden: (num_layers * num_directions, batch, hidden_size)
        # out변수와 hidden변수에 연산한 결과 저장
        # hidden과 out (연산 결과) 리턴 / out.view(-1, num_classes)는 output을 N * 5 shape로 하기 위함
        out, hidden = self.rnn(x, hidden)
        return hidden, out.view(-1, num_classes)

    def init_hidden(self):
        # Initialize hidden and cell states
        # (num_layers * num_directions, batch, hidden_size)
        # 모델 처음 시작시 hidden zero 배열로 초기화
        return Variable(torch.zeros(num_layers, batch_size, hidden_size))


# Instantiate RNN model
# Model 클래스 변수 model 선언
# Model((rnn): RNN(5, 5, batch_first=True))
model = Model()
print(model)

# Set loss and optimizer function
# CrossEntropyLoss = LogSoftmax + NLLLoss
# 손실함수로 CrossEntropy 사용
# Adam 사용, model.parameters()를 이용하여 parameter를 전달 학습률 0.1
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# Train the model
# 100번 반복 학습
for epoch in range(100):
    # 미분을 통해 얻은 기울기가 이전에 계산된 기울기 값에 누적되기 때문에 기울기 값을 0으로 초기화
    # loss값 초기화
    # hidden 초기화
    optimizer.zero_grad()
    loss = 0
    hidden = model.init_hidden()

    # 인터프리터 설정(>>>대신)
    sys.stdout.write("predicted string: ")
    # for을 이용하여 loss값 도출
    for input, label in zip(inputs, labels):
        # print(input.size(), label.size())
        # hidden변수와 output 변수에 연산한 결과 저장
        # output.max(1)을 통해 output에서 큰 값을 찾음 (무엇을 예측했는지 확인하기 위함)
        # 무엇을 예측했는지 char 출력
        # loss값 연산 후 저장
        hidden, output = model(hidden, input)
        val, idx = output.max(1)
        sys.stdout.write(idx2char[idx.data[0]])
        loss += criterion(output, torch.LongTensor([label]))

    print(", epoch: %d, loss: %1.3f" % (epoch + 1, loss))

    # 역전파 실행, 손실 함수를 미분하여 기울기 계산
    # step() 함수를 호출하여 parameter를 업데이트함
    loss.backward()
    optimizer.step()

print("Learning finished!")
