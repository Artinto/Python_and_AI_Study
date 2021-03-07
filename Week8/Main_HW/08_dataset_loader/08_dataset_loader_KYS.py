from torch.utils.data import Dataset, DataLoader # Dataset, DataLoader import
from torch import from_numpy, tensor # from_numpy, tensor import
import numpy as np # numpy import and call np

class DiabetesDataset(Dataset): # Dataset을 상속받는 DiabetesDataset 클래스
    """ Diabetes dataset."""

    # Initialize your data, download, etc.
    def __init__(self): # constructor
        xy = np.loadtxt('./data/diabetes.csv.gz',
                        delimiter=',', dtype=np.float32) # numpy를 활용하여 파일 불러오기 (데이터는 ','을 기준으로 나누고 형식은 float32)
        self.len = xy.shape[0]    # 파일 길이 저장
        self.x_data = from_numpy(xy[:, 0:-1]) # 0열 ~ 마지막 -1 열까지 저장
        self.y_data = from_numpy(xy[:, [-1]]) # 마지막 열 저장

    def __getitem__(self, index): # index의 맞는 x_data, y_data값 반환
        return self.x_data[index], self.y_data[index]

    def __len__(self): # xy(파일)의 길이 반환
        return self.len


dataset = DiabetesDataset() # 인스턴스 할당
train_loader = DataLoader(dataset=dataset,
                          batch_size=32,
                          shuffle=True,  # 데이터를 섞는다
                          num_workers=2) # 스레드 2개 사용

for epoch in range(2): # 총 2번의 학습
    for i, data in enumerate(train_loader, 0): # 1 loop == batch_size(32)
        # get the inputs
        inputs, labels = data # inputs : x_data, labels = y_data

        # wrap them in Variable
        inputs, labels = tensor(inputs), tensor(labels) # tensor로 변형

        # Run your training process
        print(f'Epoch: {i} | Inputs {inputs.data} | Labels {labels.data}')