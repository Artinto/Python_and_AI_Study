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
    
    y_pred_After_sigmoid = y_pred_before_sigmoid
    return y_pred_After_sigmoid
        
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

"""

batch :  1 loss:  0.7153223752975464
batch :  2 loss:  0.5730443596839905
batch :  3 loss:  0.661115825176239
batch :  4 loss:  0.6412070393562317
batch :  5 loss:  0.6605209708213806
batch :  6 loss:  0.6239943504333496
batch :  7 loss:  0.6225529909133911
batch :  8 loss:  0.7164117097854614
batch :  9 loss:  0.6970505714416504
batch :  10 loss:  0.6941198110580444
batch :  11 loss:  0.5572051405906677
batch :  12 loss:  0.6424752473831177
batch :  13 loss:  0.6786003112792969
batch :  14 loss:  0.6414138078689575
batch :  15 loss:  0.6414481401443481
batch :  16 loss:  0.6448468565940857
-------- loss per 100 Epoch ----------
 ## Epoch 100 loss:  0.6448468565940857
batch :  1 loss:  0.643223226070404
batch :  2 loss:  0.5942999720573425
batch :  3 loss:  0.6549132466316223
batch :  4 loss:  0.6113916039466858
batch :  5 loss:  0.6147060990333557
batch :  6 loss:  0.6062695384025574
batch :  7 loss:  0.6135531067848206
batch :  8 loss:  0.6759259700775146
batch :  9 loss:  0.5936850309371948
batch :  10 loss:  0.583781898021698
batch :  11 loss:  0.6034705638885498
batch :  12 loss:  0.5025320053100586
batch :  13 loss:  0.6230711340904236
batch :  14 loss:  0.7350099682807922
batch :  15 loss:  0.6592652201652527
batch :  16 loss:  0.6366538405418396
-------- loss per 100 Epoch ----------
 ## Epoch 200 loss:  0.6366538405418396
batch :  1 loss:  0.5099302530288696
batch :  2 loss:  0.6433303356170654
batch :  3 loss:  0.44485270977020264
batch :  4 loss:  0.4343595504760742
batch :  5 loss:  0.4478577971458435
batch :  6 loss:  0.4202921986579895
batch :  7 loss:  0.42509138584136963
batch :  8 loss:  0.4325353503227234
batch :  9 loss:  0.4745917320251465
batch :  10 loss:  0.4653199315071106
batch :  11 loss:  0.5798049569129944
batch :  12 loss:  0.47562551498413086
batch :  13 loss:  0.477874755859375
batch :  14 loss:  0.651991069316864
batch :  15 loss:  0.5797448754310608
batch :  16 loss:  0.5101121664047241
-------- loss per 100 Epoch ----------
 ## Epoch 300 loss:  0.5101121664047241
batch :  1 loss:  0.5797200798988342
batch :  2 loss:  0.549056887626648
batch :  3 loss:  0.3748357892036438
batch :  4 loss:  0.37850645184516907
batch :  5 loss:  0.4812867343425751
batch :  6 loss:  0.5350621938705444
batch :  7 loss:  0.43841081857681274
batch :  8 loss:  0.4512312412261963
batch :  9 loss:  0.5421929955482483
batch :  10 loss:  0.4881836771965027
batch :  11 loss:  0.5104740262031555
batch :  12 loss:  0.5587791204452515
batch :  13 loss:  0.5896449089050293
batch :  14 loss:  0.4239252209663391
batch :  15 loss:  0.4959260821342468
batch :  16 loss:  0.4496863782405853
-------- loss per 100 Epoch ----------
 ## Epoch 400 loss:  0.4496863782405853
batch :  1 loss:  0.3753618896007538
batch :  2 loss:  0.4857098460197449
batch :  3 loss:  0.64762282371521
batch :  4 loss:  0.37156304717063904
batch :  5 loss:  0.5186015367507935
batch :  6 loss:  0.7126872539520264
batch :  7 loss:  0.5189546942710876
batch :  8 loss:  0.39423784613609314
batch :  9 loss:  0.4609542489051819
batch :  10 loss:  0.5317912101745605
batch :  11 loss:  0.44077152013778687
batch :  12 loss:  0.4075796902179718
batch :  13 loss:  0.5274664759635925
batch :  14 loss:  0.4730958938598633
batch :  15 loss:  0.5165308713912964
batch :  16 loss:  0.4316021502017975
-------- loss per 100 Epoch ----------
 ## Epoch 500 loss:  0.4316021502017975
batch :  1 loss:  0.48097872734069824
batch :  2 loss:  0.5838167667388916
batch :  3 loss:  0.4402921795845032
batch :  4 loss:  0.6069121956825256
batch :  5 loss:  0.43418774008750916
batch :  6 loss:  0.4427271783351898
batch :  7 loss:  0.5666354298591614
batch :  8 loss:  0.6505724191665649
batch :  9 loss:  0.47937479615211487
batch :  10 loss:  0.418759286403656
batch :  11 loss:  0.48552781343460083
batch :  12 loss:  0.4682575762271881
batch :  13 loss:  0.4479965567588806
batch :  14 loss:  0.4118937849998474
batch :  15 loss:  0.37467604875564575
batch :  16 loss:  0.5463704466819763
-------- loss per 100 Epoch ----------
 ## Epoch 600 loss:  0.5463704466819763
batch :  1 loss:  0.6367530822753906
batch :  2 loss:  0.45669615268707275
batch :  3 loss:  0.6225671172142029
batch :  4 loss:  0.5314328670501709
batch :  5 loss:  0.40183523297309875
batch :  6 loss:  0.373384952545166
batch :  7 loss:  0.4804024398326874
batch :  8 loss:  0.45627835392951965
batch :  9 loss:  0.4527077376842499
batch :  10 loss:  0.45181217789649963
batch :  11 loss:  0.580909788608551
batch :  12 loss:  0.446627140045166
batch :  13 loss:  0.4774783253669739
batch :  14 loss:  0.45722705125808716
batch :  15 loss:  0.5813412070274353
batch :  16 loss:  0.378853976726532
-------- loss per 100 Epoch ----------
 ## Epoch 700 loss:  0.378853976726532
batch :  1 loss:  0.7490993142127991
batch :  2 loss:  0.42834869027137756
batch :  3 loss:  0.31264299154281616
batch :  4 loss:  0.46987345814704895
batch :  5 loss:  0.5831025838851929
batch :  6 loss:  0.5593169331550598
batch :  7 loss:  0.5486868023872375
batch :  8 loss:  0.511567234992981
batch :  9 loss:  0.469098836183548
batch :  10 loss:  0.41662582755088806
batch :  11 loss:  0.4907402992248535
batch :  12 loss:  0.33428481221199036
batch :  13 loss:  0.47657153010368347
batch :  14 loss:  0.3726765513420105
batch :  15 loss:  0.660319447517395
batch :  16 loss:  0.392945796251297
-------- loss per 100 Epoch ----------
 ## Epoch 800 loss:  0.392945796251297
batch :  1 loss:  0.35432177782058716
batch :  2 loss:  0.3250982463359833
batch :  3 loss:  0.5627031326293945
batch :  4 loss:  0.5268199443817139
batch :  5 loss:  0.42795702815055847
batch :  6 loss:  0.6722607612609863
batch :  7 loss:  0.5039638876914978
batch :  8 loss:  0.48793214559555054
batch :  9 loss:  0.4000788629055023
batch :  10 loss:  0.5686221718788147
batch :  11 loss:  0.5200484991073608
batch :  12 loss:  0.5059975385665894
batch :  13 loss:  0.4889884889125824
batch :  14 loss:  0.3995327055454254
batch :  15 loss:  0.6175566911697388
batch :  16 loss:  0.4141431450843811
-------- loss per 100 Epoch ----------
 ## Epoch 900 loss:  0.4141431450843811
batch :  1 loss:  0.4025745689868927
batch :  2 loss:  0.4223707914352417
batch :  3 loss:  0.5884566903114319
batch :  4 loss:  0.4570043981075287
batch :  5 loss:  0.5342998504638672
batch :  6 loss:  0.5051179528236389
batch :  7 loss:  0.4197564423084259
batch :  8 loss:  0.4989599883556366
batch :  9 loss:  0.3994888365268707
batch :  10 loss:  0.4948764443397522
batch :  11 loss:  0.5205739736557007
batch :  12 loss:  0.4182160794734955
batch :  13 loss:  0.42590275406837463
batch :  14 loss:  0.4827919006347656
batch :  15 loss:  0.714404284954071
batch :  16 loss:  0.42681294679641724
-------- loss per 100 Epoch ----------
 ## Epoch 1000 loss:  0.42681294679641724
Accuracy:  78.76447876447877 %

"""
