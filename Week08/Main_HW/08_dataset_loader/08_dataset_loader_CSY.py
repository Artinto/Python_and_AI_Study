# References
# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/pytorch_basics/main.py
# http://pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class
# Dataset, DataLoader를 불러와 미니배치 단위로 처리할 수 있도록 함
# tensor와 from_numpy를 불러옴. torch.from_numpy는 자동으로 input array의 dtype을 상속받고 tensor와 메모리 버퍼를 공유함. tensor 값이 변경되면 Numpy array값 변경됨.
# numpy 라이브러리를 불러옴
from torch.utils.data import Dataset, DataLoader
from torch import from_numpy, tensor
import numpy as np

# Dataset class를 상속받는 파이썬 클래스 정의
class DiabetesDataset(Dataset):
    """ Diabetes dataset."""

    # Initialize your data, download, etc.
    # 모델의 생성자 정의 (객체가 갖는 속성값을 초기화하는 역할, 객체가 생성될 때 자동으로 호출됨)
    def __init__(self):
        # xy에 데이터를 로드하고 ','로 구분하며 데이터 타입을 np.float32로 설정
        # xy 데이터의 길이 계산
        # x_data는 데이터에서 0번째부터 끝에서 2번째까지의 데이터를, y_data는 마지막 데이터를 말하며 tensor로 변환할 때 원래 메모리를 상속받음
        xy = np.loadtxt('./data/diabetes.csv.gz',
                        delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = from_numpy(xy[:, 0:-1])
        self.y_data = from_numpy(xy[:, [-1]])

    # 모델 객체와 index를 받아 해당 x_data와 y_data를 반환
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # 모델 객체를 받아 데이터 길이 반환
    def __len__(self):
        return self.len

# DiabetesDataset 클래스를 dataset으로 인스턴스화
# train_loader에 배치 사이즈가 32이며 순서가 무작위인 데이터를 로드함 (num_workers는 학습 도중 CPU의 작업을 몇 개의 코어를 사용해서 진행할지에 대한 설정 파라미터임)
dataset = DiabetesDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=2)

# 2번 반복
# 전체 데이터인 train_loader를 넘겨줌 (enumerate 함수는 순서와 리스트의 값을 전달함)
for epoch in range(2):
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        # data를 inputs과 labels로 나눔
        inputs, labels = data

        # wrap them in Variable
        # inputs와 lables의 데이터를 tensor로 변환
        inputs, labels = tensor(inputs), tensor(labels)

        # Run your training process
        # 반복 횟수와 inputs값과 labels값을 차례로 출력
        print(f'Epoch: {i} | Inputs {inputs.data} | Labels {labels.data}')
