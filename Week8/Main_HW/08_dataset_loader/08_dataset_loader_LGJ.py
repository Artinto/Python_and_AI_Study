# References
# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/pytorch_basics/main.py
# http://pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class
from torch.utils.data import Dataset, DataLoader
from torch import from_numpy, tensor
import numpy as np

class DiabetesDataset(Dataset): # Dataset 클래스의 상속을 받는 DiabetesDataset 클래서 생성
    """ Diabetes dataset."""

    # Initialize your data, download, etc.
    def __init__(self): # 초기화 매서드
        xy = np.loadtxt('./data/diabetes.csv.gz',
                        delimiter=',', dtype=np.float32)  # xy의 데이터를 넘파이형식으로 로드함. 경로'./data/diabetes.csv.gz', 구분자 ',' 
        self.len = xy.shape[0]  # 데이터의 길이, 즉 xy행렬의 행
        self.x_data = from_numpy(xy[:, 0:-1]) # xy 데이터셋의 맨 마지막 열을 제하고 나머지를 x_data로 지정
        self.y_data = from_numpy(xy[:, [-1]]) # xy 데이터셋의 맨 마지막 열만 y_data로 지정

    def __getitem__(self, index): # index의 항목을 반환하는 매서드
        return self.x_data[index], self.y_data[index] 

    def __len__(self):  # 데이터의 길이반환 매서드
        return self.len


dataset = DiabetesDataset() # DiabetesDataset의 인스턴스로 dataset을 만듦
train_loader = DataLoader(dataset=dataset,
                          batch_size=32,  # 배치 사이즈 32
                          shuffle=True, # 데이터 셔플 사용
                          num_workers=2)  # 다양한 처리를 위해 사용

for epoch in range(2):
    for i, data in enumerate(train_loader, 0):  # dataloader를 통해 불러온 1 iteration
        # get the inputs
        inputs, labels = data # 배치 데이터를 input과 labels로 나눔

        # wrap them in Variable
        inputs, labels = tensor(inputs), tensor(labels) # inputs, labels를 tensor로 만듦

        # Run your training process
        print(f'Epoch: {i} | Inputs {inputs.data} | Labels {labels.data}')  # Epoch, Inputs, Labels 개수를 각각 출력
