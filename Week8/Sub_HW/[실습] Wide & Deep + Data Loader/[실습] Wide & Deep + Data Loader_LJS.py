from torch.utils.data import Dataset, DataLoader
import numpy as np 
import torch

Data = np.loadtxt("diabetes.csv", delimiter=',', dtype = np.float32)
x_train = Data[:500, :-1]
y_train = Data[:500, [-1]]
x_test  = Data[500:, :-1]
y_test  = Data[500:, [-1]]

# Dataloader Class 정의

class DiabetesDataset(Dataset):
    def __init__(self):
        self.x_train = torch.from_numpy(x_train)  # type : numpy -> tensor
        self.y_train = torch.from_numpy(y_train)
        
        self.length = x_train.shape[0] # Data 759 x 8 , shape[0] = 759
        
    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]
    
    def __len__(self):
        return self.length
    
dataset = DiabetesDataset()

train_loader = DataLoader(dataset = dataset, batch_size = 32, shuffle = True, num_workers = 0)

class Model(torch.nn.Module):
    
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = torch.nn.Linear(8,6)
        self.l2 = torch.nn.Linear(6,4)
        self.l3 = torch.nn.Linear(4,1)
        
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self,x):
        out1 = self.sigmoid(self.l1(x))
        out2 = self.sigmoid(self.l2(out1))
        y_pred = self.sigmoid(self.l3(out2))
        
        return y_pred
    
model = Model()

criterion = torch.nn.BCELoss(reduction = "mean")
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)

def Accuracy(y_pred, y_test):
    acc = 0 
    for i in range(len(y_test)):
        if(y_pred[i] == y_test[i]):
            acc += 1 
    return (acc/len(y_pred)) * 100 
    
def predict(y_pred_before_sigmoid):
    
    for i in range(len(y_pred_before_sigmoid)):
        
        if(y_pred_before_sigmoid[i] > 0.5):
            y_pred_before_sigmoid[i] = 1
        else:
            y_pred_before_sigmoid[i] = 0
    
    return y_pred_before_sigmoid
        
batch = 0

# 총 데이터 1000번 학습 ->>>> ( 총 데이터를 데이터 32개로 나누어서 학습 ) x 1000번 
for epoch in range(1001):
    batch = 0 
    
    # Batch size 32
    for i, data in enumerate(train_loader, 0):
        batch += 1 # batch number 
        inputs, labels = data 
        
        # print("data : ", data) >> inputs : 32x8 , labels : 32x1 
        # print("type(inputs)1", type(inputs))  이미 tensor type 

        y_pred = model(inputs)
        loss = criterion(y_pred, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if( (epoch+1) % 100 == 0 ):
            print("batch : ", batch, "loss: ", loss.item() )
    if( (epoch+1) % 100 == 0 ):
        print("-------- loss per 100 Epoch ----------")
        print(" ## Epoch", epoch+1, "loss: " , loss.item())
        
        
x_test = torch.tensor(x_test)        # Tensor type 으로 변환
y_test = torch.tensor(y_test)        # Tensor type 으로 변환
prediction = predict( model(x_test) ) # wide and deep 하게 Layer를 통과하여 나온 값을 Sigmoid 함수를 통해 Binary Classification 

print("Accuracy: ", Accuracy(prediction, y_test) , "%" ) # (맞 개수 / 전체 개수)  * 100
