from torch.utils.data import Dataset, DataLoader
from torch import from_numpy, tensor
import numpy as np

class DiabetesDataset(Dataset):
    """ Diabetes dataset."""

    # Initialize your data, download, etc.
    def __init__(self):
        xy = np.loadtxt('./data/diabetes.csv.gz',
                        delimiter=',', dtype=np.float32) #데이터셋 로드
        self.len = xy.shape[0] #데이터셋의 크기
        self.x_data = from_numpy(xy[:, 0:-1]) #numpy배열 -> tensor
        self.y_data = from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index] #index번째 데이터 반환

    def __len__(self):
        return self.len #데이터셋의 크기 리턴


dataset = DiabetesDataset()
train_loader = DataLoader(dataset=dataset, 
                          batch_size=32,
                          shuffle=True,
                          num_workers=2) #2개의 서브 프로세싱 사용

for epoch in range(2):
    for i, data in enumerate(train_loader, 0): #각 epoch마다의 반복자
        # get the inputs
        inputs, labels = data #batch데이터를 인풋과 라벨로 나눔

        # wrap them in Variable
        inputs, labels = tensor(inputs), tensor(labels) #텐서로 변환

        # Run your training process
        print(f'Epoch: {i} | Inputs {inputs.data} | Labels {labels.data}')
