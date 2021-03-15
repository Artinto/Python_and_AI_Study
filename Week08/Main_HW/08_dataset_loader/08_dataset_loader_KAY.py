from torch.utils.data import Dataset, DataLoader
from torch import from_numpy, tensor
import numpy as np

class DiabetesDataset(Dataset):#Dataset을 가져오는 클라쓰
    """ Diabetes dataset."""

    # Initialize your data, download, etc.
    def __init__(self):
        xy = np.loadtxt('./data/diabetes.csv.gz',
                        delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = from_numpy(xy[:, 0:-1])#마지막 항 제외하고 저장 
        self.y_data = from_numpy(xy[:, [-1]])#마지막 항 저장

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


dataset = DiabetesDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=2)#멀티쓰레딩을 2로 

for epoch in range(2):
    for i, data in enumerate(train_loader, 0):# 인덱스를 붙인다 즉 각 epoch마다 같은 인덱스를 갖는다
        # get the inputs
        inputs, labels = data#batch 데이터를 나눔

        # wrap them in Variable
        inputs, labels = tensor(inputs), tensor(labels)#텐서로 

        # Run your training process
        print(f'Epoch: {i} | Inputs {inputs.data} | Labels {labels.data}')
