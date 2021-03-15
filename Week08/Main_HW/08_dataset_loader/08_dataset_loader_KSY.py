
# References
# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/pytorch_basics/main.py
# http://pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class
from torch.utils.data import Dataset, DataLoader
from torch import from_numpy, tensor
import numpy as np

class DiabetesDataset(Dataset):
    """ Diabetes dataset."""

    # Initialize your data, download, etc.
    def __init__(self):
        xy = np.loadtxt('./diabetes.csv', delimiter=',', dtype=np.float32) # 데이터 가져오기
        self.len = xy.shape[0]
        self.x_data = from_numpy(xy[:, 0:-1])
        self.y_data = from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index] #인덱스 순서의 데이터 가져오기

    def __len__(self):
        return self.len #길이


dataset = DiabetesDataset()
train_loader = DataLoader(dataset=dataset,batch_size=32, shuffle=True,num_workers=2) # 데이터셋을 32크기로 로드하고 셔플한후 2번반복

for epoch in range(2):
    for i, data in enumerate(train_loader, 0): #인덱스= epoch
        # get the inputs
        inputs, labels = data#

        # wrap them in Variable
        inputs, labels = tensor(inputs), tensor(labels) #  텐서

        # Run your training process
        print(f'Epoch: {i} | Inputs {inputs.data} | Labels {labels.data}')
