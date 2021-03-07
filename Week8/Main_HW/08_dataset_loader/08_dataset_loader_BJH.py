from torch.utils.data import Dataset, DataLoader
from torch import from_numpy, tensor
import numpy as np



class DiabetesDataset(Dataset):
    """ Diabetes dataset."""

    # Initialize your data, download, etc.
    def __init__(self):
        xy = np.loadtxt('./data/diabetes.csv',
                        delimiter=',', dtype=np.float32) #문서 불러오기
        self.len = xy.shape[0] #행의 크기
        self.x_data = from_numpy(xy[:, 0:-1])
        self.y_data = from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index] #index로 입력받은 변수의 값 출력

    def __len__(self):
        return self.len #데이터의 크기 리턴



dataset = DiabetesDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=32, #한번에 넘겨주는 데이터의 수를 의미
                          shuffle=True,
                          num_workers=0)

for epoch in range(2):
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data #데이터를 인풋과 라벨에 할당

        # wrap them in Variable
        inputs, labels = tensor(inputs), tensor(labels) #텐서로 변환

        # Run your training process
        print(f'Epoch: {i} | Inputs {inputs.data} | Labels {labels.data}')
