from torch.utils.data import Dataset, DataLoader
from torch import from_numpy, tensor
import numpy as np

# custom 데이터셋 만들기
class DiabetesDataset(Dataset):
    def __init__(self):
        # 데이터 불러오기, 변수나 데이터 선언하기
        xy = np.loadtxt('/content/drive/MyDrive/diabetes.csv',delimiter=',', dtype=np.float32)
        self.len = xy.shape[0] 
        self.x_data = from_numpy(xy[:, 0:-1]) # 불러온 데이터를 x, y데이터 나눠 선언해줌.
        self.y_data = from_numpy(xy[:, [-1]]) # class 내부에서 사용가능 

    def __getitem__(self, index):
        # __getitem__ : dataset[i]을 했을 때 i번째 샘플을 가져오도록 하는 인덱싱을 위한 것
        # 인덱스에 해당하는 x, y데이터 짝 맞추어 return해주기
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


dataset = DiabetesDataset()
train_loader = DataLoader(dataset=dataset,batch_size=32,shuffle=True, num_workers=2)
# 출력(train_loader) : x_data, y_data

# DataLoader를 통해 batch_size만큼 데이터를 나눠서 처리함.
# num_workers : cpu에서 데이터 전처리하는 작업을 빠르게 처리하기 위해(+ GPU를 놀지 않게 만들기 위해)
# task를 GPU로 던져서 GPU 사용률을 최대로 끌어내기 위한 방식으로 
for epoch in range(2): # epoch : 전체 반복
    for i, data in enumerate(train_loader, 0): # __getitem__가 여기서 실행되는 듯. / batch size만큼 돌아감.
        inputs, labels = data  # data는 list형태임
        
        inputs, labels = tensor(inputs), tensor(labels) # 이미 텐서형이라서 안해도 될거 같긴함.
        
        print(f'Epoch: {i} | Inputs {inputs.data} | Labels {labels.data}')
