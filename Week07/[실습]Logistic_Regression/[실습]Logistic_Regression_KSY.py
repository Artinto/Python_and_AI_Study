import pandas as pd
import numpy as np
from torch import nn
import torch
from torch import tensor
from torch import sigmoid
import torch.nn.functional as F
import torch.optim as optim
x=np.loadtxt("./diabetes.csv",usecols=(0,1,2,3,4,5,6,7),delimiter=',',dtype=np.float32)
y=np.loadtxt("./diabetes.csv",usecols=(8),delimiter=',',dtype=np.float32)
x_data = torch.FloatTensor(x)
y_data = torch.FloatTensor(y)
y_data = y_data.unsqueeze(-1)
class Model(nn.Module): #Module 상위클라스 Model 하위클라스
    def __init__(self):
       
        super(Model, self).__init__() #self를 안 써도됨 이것도 생성자 초기화 방법
        self.linear = nn.Linear(8, 1)  # One in and one out 8인풋  1아웃풋

    def forward(self, x):
        
        y_pred = sigmoid(self.linear(x)) 
        
        return y_pred
model = Model() 
criterion = nn.BCELoss(reduction='mean') #BCE (분류기) 오차를 대입
optimizer = optim.SGD(model.parameters(), lr=0.01)
def predict(x):
    linear_=nn.Linear(8,1)
    
    y_pred = sigmoid(linear_(x))
    
    result = 0
    
    if y_pred > 0.5 :
        result = 1
    else:
        result = 0
        
    return result

def accuracy(x, y):
    ACC = 0
    for x_,y_ in zip(x,y):
        if predict(x_) == y_:
            ACC += 1
    result = ( ACC/len(x) ) * 100
    return result
        
for epoch in range(1000):#1000번 반복
    y_pred = model(x_data) #model 생성자에 forward값을 대입
    loss = criterion(y_pred, y_data)  #손실loss 를 구함(분류기사용)
    print(f'Epoch {epoch + 1}/1000 | Loss: {loss.item():.4f}')
    optimizer.zero_grad()#변화율 0으로 초기화
    loss.backward() #loss초기화
    optimizer.step() #업데이트
    
print(f'\nLet\'s predict the hours need to score above 50%\n{"=" * 50}')
hour_var = model(torch.Tensor([[-0.353535,0.452353,0,0.125423,0.233515,0.266574,-0.215632,-0.022222]]))
print(f'Prediction after 1 hours of training: {hour_var.item():.4f} | Above 50%: { hour_var.item() > 0.5}')
print("Accuracy : ", accuracy(x_data,y_data), "%")



Epoch 1/1000 | Loss: 0.7240
Epoch 2/1000 | Loss: 0.7227
Epoch 3/1000 | Loss: 0.7214
Epoch 4/1000 | Loss: 0.7202
Epoch 5/1000 | Loss: 0.7189
Epoch 6/1000 | Loss: 0.7176
Epoch 7/1000 | Loss: 0.7164
Epoch 8/1000 | Loss: 0.7152
Epoch 9/1000 | Loss: 0.7140
Epoch 10/1000 | Loss: 0.7128
Epoch 11/1000 | Loss: 0.7116
Epoch 12/1000 | Loss: 0.7104
Epoch 13/1000 | Loss: 0.7093
Epoch 14/1000 | Loss: 0.7081
Epoch 15/1000 | Loss: 0.7070
Epoch 16/1000 | Loss: 0.7059
Epoch 17/1000 | Loss: 0.7047
Epoch 18/1000 | Loss: 0.7036
Epoch 19/1000 | Loss: 0.7025
Epoch 20/1000 | Loss: 0.7015
Epoch 21/1000 | Loss: 0.7004
Epoch 22/1000 | Loss: 0.6993
Epoch 23/1000 | Loss: 0.6983
Epoch 24/1000 | Loss: 0.6973
Epoch 25/1000 | Loss: 0.6962
Epoch 26/1000 | Loss: 0.6952
Epoch 27/1000 | Loss: 0.6942
Epoch 28/1000 | Loss: 0.6932
Epoch 29/1000 | Loss: 0.6922
Epoch 30/1000 | Loss: 0.6913
Epoch 31/1000 | Loss: 0.6903
Epoch 32/1000 | Loss: 0.6893
Epoch 33/1000 | Loss: 0.6884
Epoch 34/1000 | Loss: 0.6875
Epoch 35/1000 | Loss: 0.6865
Epoch 36/1000 | Loss: 0.6856
Epoch 37/1000 | Loss: 0.6847
Epoch 38/1000 | Loss: 0.6838
Epoch 39/1000 | Loss: 0.6829
Epoch 40/1000 | Loss: 0.6821
Epoch 41/1000 | Loss: 0.6812
Epoch 42/1000 | Loss: 0.6803
Epoch 43/1000 | Loss: 0.6795
Epoch 44/1000 | Loss: 0.6786
Epoch 45/1000 | Loss: 0.6778
Epoch 46/1000 | Loss: 0.6770
Epoch 47/1000 | Loss: 0.6762
Epoch 48/1000 | Loss: 0.6754
Epoch 49/1000 | Loss: 0.6746
Epoch 50/1000 | Loss: 0.6738
Epoch 51/1000 | Loss: 0.6730
Epoch 52/1000 | Loss: 0.6722
Epoch 53/1000 | Loss: 0.6714
Epoch 54/1000 | Loss: 0.6707
Epoch 55/1000 | Loss: 0.6699
Epoch 56/1000 | Loss: 0.6692
Epoch 57/1000 | Loss: 0.6684
Epoch 58/1000 | Loss: 0.6677
Epoch 59/1000 | Loss: 0.6670
Epoch 60/1000 | Loss: 0.6663
Epoch 61/1000 | Loss: 0.6656
Epoch 62/1000 | Loss: 0.6649
Epoch 63/1000 | Loss: 0.6642
Epoch 64/1000 | Loss: 0.6635
Epoch 65/1000 | Loss: 0.6628
Epoch 66/1000 | Loss: 0.6621
Epoch 67/1000 | Loss: 0.6615
Epoch 68/1000 | Loss: 0.6608
Epoch 69/1000 | Loss: 0.6601
Epoch 70/1000 | Loss: 0.6595
Epoch 71/1000 | Loss: 0.6589
Epoch 72/1000 | Loss: 0.6582
Epoch 73/1000 | Loss: 0.6576
Epoch 74/1000 | Loss: 0.6570
Epoch 75/1000 | Loss: 0.6564
Epoch 76/1000 | Loss: 0.6557
Epoch 77/1000 | Loss: 0.6551
Epoch 78/1000 | Loss: 0.6545
Epoch 79/1000 | Loss: 0.6540
Epoch 80/1000 | Loss: 0.6534
Epoch 81/1000 | Loss: 0.6528
Epoch 82/1000 | Loss: 0.6522
Epoch 83/1000 | Loss: 0.6516
Epoch 84/1000 | Loss: 0.6511
Epoch 85/1000 | Loss: 0.6505
Epoch 86/1000 | Loss: 0.6500
Epoch 87/1000 | Loss: 0.6494
Epoch 88/1000 | Loss: 0.6489
Epoch 89/1000 | Loss: 0.6484
Epoch 90/1000 | Loss: 0.6478
Epoch 91/1000 | Loss: 0.6473
Epoch 92/1000 | Loss: 0.6468
Epoch 93/1000 | Loss: 0.6463
Epoch 94/1000 | Loss: 0.6458
Epoch 95/1000 | Loss: 0.6452
Epoch 96/1000 | Loss: 0.6447
Epoch 97/1000 | Loss: 0.6443
Epoch 98/1000 | Loss: 0.6438
Epoch 99/1000 | Loss: 0.6433
Epoch 100/1000 | Loss: 0.6428
Epoch 101/1000 | Loss: 0.6423
Epoch 102/1000 | Loss: 0.6418
Epoch 103/1000 | Loss: 0.6414
Epoch 104/1000 | Loss: 0.6409
Epoch 105/1000 | Loss: 0.6405
Epoch 106/1000 | Loss: 0.6400
Epoch 107/1000 | Loss: 0.6396
Epoch 108/1000 | Loss: 0.6391
Epoch 109/1000 | Loss: 0.6387
Epoch 110/1000 | Loss: 0.6382
Epoch 111/1000 | Loss: 0.6378
Epoch 112/1000 | Loss: 0.6374
Epoch 113/1000 | Loss: 0.6369
Epoch 114/1000 | Loss: 0.6365
Epoch 115/1000 | Loss: 0.6361
Epoch 116/1000 | Loss: 0.6357
Epoch 117/1000 | Loss: 0.6353
Epoch 118/1000 | Loss: 0.6349
Epoch 119/1000 | Loss: 0.6345
Epoch 120/1000 | Loss: 0.6341
Epoch 121/1000 | Loss: 0.6337
Epoch 122/1000 | Loss: 0.6333
Epoch 123/1000 | Loss: 0.6329
Epoch 124/1000 | Loss: 0.6325
Epoch 125/1000 | Loss: 0.6321
Epoch 126/1000 | Loss: 0.6318
Epoch 127/1000 | Loss: 0.6314
Epoch 128/1000 | Loss: 0.6310
Epoch 129/1000 | Loss: 0.6307
Epoch 130/1000 | Loss: 0.6303
Epoch 131/1000 | Loss: 0.6299
Epoch 132/1000 | Loss: 0.6296
Epoch 133/1000 | Loss: 0.6292
Epoch 134/1000 | Loss: 0.6289
Epoch 135/1000 | Loss: 0.6285
Epoch 136/1000 | Loss: 0.6282
Epoch 137/1000 | Loss: 0.6278
Epoch 138/1000 | Loss: 0.6275
Epoch 139/1000 | Loss: 0.6272
Epoch 140/1000 | Loss: 0.6268
Epoch 141/1000 | Loss: 0.6265
Epoch 142/1000 | Loss: 0.6262
Epoch 143/1000 | Loss: 0.6259
Epoch 144/1000 | Loss: 0.6256
Epoch 145/1000 | Loss: 0.6252
Epoch 146/1000 | Loss: 0.6249
Epoch 147/1000 | Loss: 0.6246
Epoch 148/1000 | Loss: 0.6243
Epoch 149/1000 | Loss: 0.6240
Epoch 150/1000 | Loss: 0.6237
Epoch 151/1000 | Loss: 0.6234
Epoch 152/1000 | Loss: 0.6231
Epoch 153/1000 | Loss: 0.6228
Epoch 154/1000 | Loss: 0.6225
Epoch 155/1000 | Loss: 0.6222
Epoch 156/1000 | Loss: 0.6219
Epoch 157/1000 | Loss: 0.6216
Epoch 158/1000 | Loss: 0.6214
Epoch 159/1000 | Loss: 0.6211
Epoch 160/1000 | Loss: 0.6208
Epoch 161/1000 | Loss: 0.6205
Epoch 162/1000 | Loss: 0.6203
Epoch 163/1000 | Loss: 0.6200
Epoch 164/1000 | Loss: 0.6197
Epoch 165/1000 | Loss: 0.6194
Epoch 166/1000 | Loss: 0.6192
Epoch 167/1000 | Loss: 0.6189
Epoch 168/1000 | Loss: 0.6187
Epoch 169/1000 | Loss: 0.6184
Epoch 170/1000 | Loss: 0.6182
Epoch 171/1000 | Loss: 0.6179
Epoch 172/1000 | Loss: 0.6176
Epoch 173/1000 | Loss: 0.6174
Epoch 174/1000 | Loss: 0.6172
Epoch 175/1000 | Loss: 0.6169
Epoch 176/1000 | Loss: 0.6167
Epoch 177/1000 | Loss: 0.6164
Epoch 178/1000 | Loss: 0.6162
Epoch 179/1000 | Loss: 0.6159
Epoch 180/1000 | Loss: 0.6157
Epoch 181/1000 | Loss: 0.6155
Epoch 182/1000 | Loss: 0.6152
Epoch 183/1000 | Loss: 0.6150
Epoch 184/1000 | Loss: 0.6148
Epoch 185/1000 | Loss: 0.6146
Epoch 186/1000 | Loss: 0.6143
Epoch 187/1000 | Loss: 0.6141
Epoch 188/1000 | Loss: 0.6139
Epoch 189/1000 | Loss: 0.6137
Epoch 190/1000 | Loss: 0.6135
Epoch 191/1000 | Loss: 0.6132
Epoch 192/1000 | Loss: 0.6130
Epoch 193/1000 | Loss: 0.6128
Epoch 194/1000 | Loss: 0.6126
Epoch 195/1000 | Loss: 0.6124
Epoch 196/1000 | Loss: 0.6122
Epoch 197/1000 | Loss: 0.6120
Epoch 198/1000 | Loss: 0.6118
Epoch 199/1000 | Loss: 0.6116
Epoch 200/1000 | Loss: 0.6114
Epoch 201/1000 | Loss: 0.6112
Epoch 202/1000 | Loss: 0.6110
Epoch 203/1000 | Loss: 0.6108
Epoch 204/1000 | Loss: 0.6106
Epoch 205/1000 | Loss: 0.6104
Epoch 206/1000 | Loss: 0.6102
Epoch 207/1000 | Loss: 0.6100
Epoch 208/1000 | Loss: 0.6098
Epoch 209/1000 | Loss: 0.6096
Epoch 210/1000 | Loss: 0.6095
Epoch 211/1000 | Loss: 0.6093
Epoch 212/1000 | Loss: 0.6091
Epoch 213/1000 | Loss: 0.6089
Epoch 214/1000 | Loss: 0.6087
Epoch 215/1000 | Loss: 0.6086
Epoch 216/1000 | Loss: 0.6084
Epoch 217/1000 | Loss: 0.6082
Epoch 218/1000 | Loss: 0.6080
Epoch 219/1000 | Loss: 0.6078
Epoch 220/1000 | Loss: 0.6077
Epoch 221/1000 | Loss: 0.6075
Epoch 222/1000 | Loss: 0.6073
Epoch 223/1000 | Loss: 0.6072
Epoch 224/1000 | Loss: 0.6070
Epoch 225/1000 | Loss: 0.6068
Epoch 226/1000 | Loss: 0.6067
Epoch 227/1000 | Loss: 0.6065
Epoch 228/1000 | Loss: 0.6063
Epoch 229/1000 | Loss: 0.6062
Epoch 230/1000 | Loss: 0.6060
Epoch 231/1000 | Loss: 0.6059
Epoch 232/1000 | Loss: 0.6057
Epoch 233/1000 | Loss: 0.6055
Epoch 234/1000 | Loss: 0.6054
Epoch 235/1000 | Loss: 0.6052
Epoch 236/1000 | Loss: 0.6051
Epoch 237/1000 | Loss: 0.6049
Epoch 238/1000 | Loss: 0.6048
Epoch 239/1000 | Loss: 0.6046
Epoch 240/1000 | Loss: 0.6045
Epoch 241/1000 | Loss: 0.6043
Epoch 242/1000 | Loss: 0.6042
Epoch 243/1000 | Loss: 0.6040
Epoch 244/1000 | Loss: 0.6039
Epoch 245/1000 | Loss: 0.6037
Epoch 246/1000 | Loss: 0.6036
Epoch 247/1000 | Loss: 0.6034
Epoch 248/1000 | Loss: 0.6033
Epoch 249/1000 | Loss: 0.6032
Epoch 250/1000 | Loss: 0.6030
Epoch 251/1000 | Loss: 0.6029
Epoch 252/1000 | Loss: 0.6027
Epoch 253/1000 | Loss: 0.6026
Epoch 254/1000 | Loss: 0.6025
Epoch 255/1000 | Loss: 0.6023
Epoch 256/1000 | Loss: 0.6022
Epoch 257/1000 | Loss: 0.6020
Epoch 258/1000 | Loss: 0.6019
Epoch 259/1000 | Loss: 0.6018
Epoch 260/1000 | Loss: 0.6016
Epoch 261/1000 | Loss: 0.6015
Epoch 262/1000 | Loss: 0.6014
Epoch 263/1000 | Loss: 0.6013
Epoch 264/1000 | Loss: 0.6011
Epoch 265/1000 | Loss: 0.6010
Epoch 266/1000 | Loss: 0.6009
Epoch 267/1000 | Loss: 0.6007
Epoch 268/1000 | Loss: 0.6006
Epoch 269/1000 | Loss: 0.6005
Epoch 270/1000 | Loss: 0.6004
Epoch 271/1000 | Loss: 0.6002
Epoch 272/1000 | Loss: 0.6001
Epoch 273/1000 | Loss: 0.6000
Epoch 274/1000 | Loss: 0.5999
Epoch 275/1000 | Loss: 0.5998
Epoch 276/1000 | Loss: 0.5996
Epoch 277/1000 | Loss: 0.5995
Epoch 278/1000 | Loss: 0.5994
Epoch 279/1000 | Loss: 0.5993
Epoch 280/1000 | Loss: 0.5992
Epoch 281/1000 | Loss: 0.5990
Epoch 282/1000 | Loss: 0.5989
Epoch 283/1000 | Loss: 0.5988
Epoch 284/1000 | Loss: 0.5987
Epoch 285/1000 | Loss: 0.5986
Epoch 286/1000 | Loss: 0.5985
Epoch 287/1000 | Loss: 0.5984
Epoch 288/1000 | Loss: 0.5982
Epoch 289/1000 | Loss: 0.5981
Epoch 290/1000 | Loss: 0.5980
Epoch 291/1000 | Loss: 0.5979
Epoch 292/1000 | Loss: 0.5978
Epoch 293/1000 | Loss: 0.5977
Epoch 294/1000 | Loss: 0.5976
Epoch 295/1000 | Loss: 0.5975
Epoch 296/1000 | Loss: 0.5974
Epoch 297/1000 | Loss: 0.5973
Epoch 298/1000 | Loss: 0.5972
Epoch 299/1000 | Loss: 0.5971
Epoch 300/1000 | Loss: 0.5969
Epoch 301/1000 | Loss: 0.5968
Epoch 302/1000 | Loss: 0.5967
Epoch 303/1000 | Loss: 0.5966
Epoch 304/1000 | Loss: 0.5965
Epoch 305/1000 | Loss: 0.5964
Epoch 306/1000 | Loss: 0.5963
Epoch 307/1000 | Loss: 0.5962
Epoch 308/1000 | Loss: 0.5961
Epoch 309/1000 | Loss: 0.5960
Epoch 310/1000 | Loss: 0.5959
Epoch 311/1000 | Loss: 0.5958
Epoch 312/1000 | Loss: 0.5957
Epoch 313/1000 | Loss: 0.5956
Epoch 314/1000 | Loss: 0.5955
Epoch 315/1000 | Loss: 0.5954
Epoch 316/1000 | Loss: 0.5953
Epoch 317/1000 | Loss: 0.5952
Epoch 318/1000 | Loss: 0.5951
Epoch 319/1000 | Loss: 0.5950
Epoch 320/1000 | Loss: 0.5949
Epoch 321/1000 | Loss: 0.5949
Epoch 322/1000 | Loss: 0.5948
Epoch 323/1000 | Loss: 0.5947
Epoch 324/1000 | Loss: 0.5946
Epoch 325/1000 | Loss: 0.5945
Epoch 326/1000 | Loss: 0.5944
Epoch 327/1000 | Loss: 0.5943
Epoch 328/1000 | Loss: 0.5942
Epoch 329/1000 | Loss: 0.5941
Epoch 330/1000 | Loss: 0.5940
Epoch 331/1000 | Loss: 0.5939
Epoch 332/1000 | Loss: 0.5938
Epoch 333/1000 | Loss: 0.5937
Epoch 334/1000 | Loss: 0.5937
Epoch 335/1000 | Loss: 0.5936
Epoch 336/1000 | Loss: 0.5935
Epoch 337/1000 | Loss: 0.5934
Epoch 338/1000 | Loss: 0.5933
Epoch 339/1000 | Loss: 0.5932
Epoch 340/1000 | Loss: 0.5931
Epoch 341/1000 | Loss: 0.5930
Epoch 342/1000 | Loss: 0.5930
Epoch 343/1000 | Loss: 0.5929
Epoch 344/1000 | Loss: 0.5928
Epoch 345/1000 | Loss: 0.5927
Epoch 346/1000 | Loss: 0.5926
Epoch 347/1000 | Loss: 0.5925
Epoch 348/1000 | Loss: 0.5924
Epoch 349/1000 | Loss: 0.5924
Epoch 350/1000 | Loss: 0.5923
Epoch 351/1000 | Loss: 0.5922
Epoch 352/1000 | Loss: 0.5921
Epoch 353/1000 | Loss: 0.5920
Epoch 354/1000 | Loss: 0.5919
Epoch 355/1000 | Loss: 0.5919
Epoch 356/1000 | Loss: 0.5918
Epoch 357/1000 | Loss: 0.5917
Epoch 358/1000 | Loss: 0.5916
Epoch 359/1000 | Loss: 0.5915
Epoch 360/1000 | Loss: 0.5915
Epoch 361/1000 | Loss: 0.5914
Epoch 362/1000 | Loss: 0.5913
Epoch 363/1000 | Loss: 0.5912
Epoch 364/1000 | Loss: 0.5911
Epoch 365/1000 | Loss: 0.5911
Epoch 366/1000 | Loss: 0.5910
Epoch 367/1000 | Loss: 0.5909
Epoch 368/1000 | Loss: 0.5908
Epoch 369/1000 | Loss: 0.5907
Epoch 370/1000 | Loss: 0.5907
Epoch 371/1000 | Loss: 0.5906
Epoch 372/1000 | Loss: 0.5905
Epoch 373/1000 | Loss: 0.5904
Epoch 374/1000 | Loss: 0.5904
Epoch 375/1000 | Loss: 0.5903
Epoch 376/1000 | Loss: 0.5902
Epoch 377/1000 | Loss: 0.5901
Epoch 378/1000 | Loss: 0.5901
Epoch 379/1000 | Loss: 0.5900
Epoch 380/1000 | Loss: 0.5899
Epoch 381/1000 | Loss: 0.5898
Epoch 382/1000 | Loss: 0.5898
Epoch 383/1000 | Loss: 0.5897
Epoch 384/1000 | Loss: 0.5896
Epoch 385/1000 | Loss: 0.5895
Epoch 386/1000 | Loss: 0.5895
Epoch 387/1000 | Loss: 0.5894
Epoch 388/1000 | Loss: 0.5893
Epoch 389/1000 | Loss: 0.5892
Epoch 390/1000 | Loss: 0.5892
Epoch 391/1000 | Loss: 0.5891
Epoch 392/1000 | Loss: 0.5890
Epoch 393/1000 | Loss: 0.5890
Epoch 394/1000 | Loss: 0.5889
Epoch 395/1000 | Loss: 0.5888
Epoch 396/1000 | Loss: 0.5887
Epoch 397/1000 | Loss: 0.5887
Epoch 398/1000 | Loss: 0.5886
Epoch 399/1000 | Loss: 0.5885
Epoch 400/1000 | Loss: 0.5885
Epoch 401/1000 | Loss: 0.5884
Epoch 402/1000 | Loss: 0.5883
Epoch 403/1000 | Loss: 0.5882
Epoch 404/1000 | Loss: 0.5882
Epoch 405/1000 | Loss: 0.5881
Epoch 406/1000 | Loss: 0.5880
Epoch 407/1000 | Loss: 0.5880
Epoch 408/1000 | Loss: 0.5879
Epoch 409/1000 | Loss: 0.5878
Epoch 410/1000 | Loss: 0.5878
Epoch 411/1000 | Loss: 0.5877
Epoch 412/1000 | Loss: 0.5876
Epoch 413/1000 | Loss: 0.5876
Epoch 414/1000 | Loss: 0.5875
Epoch 415/1000 | Loss: 0.5874
Epoch 416/1000 | Loss: 0.5874
Epoch 417/1000 | Loss: 0.5873
Epoch 418/1000 | Loss: 0.5872
Epoch 419/1000 | Loss: 0.5872
Epoch 420/1000 | Loss: 0.5871
Epoch 421/1000 | Loss: 0.5870
Epoch 422/1000 | Loss: 0.5870
Epoch 423/1000 | Loss: 0.5869
Epoch 424/1000 | Loss: 0.5868
Epoch 425/1000 | Loss: 0.5868
Epoch 426/1000 | Loss: 0.5867
Epoch 427/1000 | Loss: 0.5866
Epoch 428/1000 | Loss: 0.5866
Epoch 429/1000 | Loss: 0.5865
Epoch 430/1000 | Loss: 0.5864
Epoch 431/1000 | Loss: 0.5864
Epoch 432/1000 | Loss: 0.5863
Epoch 433/1000 | Loss: 0.5862
Epoch 434/1000 | Loss: 0.5862
Epoch 435/1000 | Loss: 0.5861
Epoch 436/1000 | Loss: 0.5861
Epoch 437/1000 | Loss: 0.5860
Epoch 438/1000 | Loss: 0.5859
Epoch 439/1000 | Loss: 0.5859
Epoch 440/1000 | Loss: 0.5858
Epoch 441/1000 | Loss: 0.5857
Epoch 442/1000 | Loss: 0.5857
Epoch 443/1000 | Loss: 0.5856
Epoch 444/1000 | Loss: 0.5856
Epoch 445/1000 | Loss: 0.5855
Epoch 446/1000 | Loss: 0.5854
Epoch 447/1000 | Loss: 0.5854
Epoch 448/1000 | Loss: 0.5853
Epoch 449/1000 | Loss: 0.5852
Epoch 450/1000 | Loss: 0.5852
Epoch 451/1000 | Loss: 0.5851
Epoch 452/1000 | Loss: 0.5851
Epoch 453/1000 | Loss: 0.5850
Epoch 454/1000 | Loss: 0.5849
Epoch 455/1000 | Loss: 0.5849
Epoch 456/1000 | Loss: 0.5848
Epoch 457/1000 | Loss: 0.5848
Epoch 458/1000 | Loss: 0.5847
Epoch 459/1000 | Loss: 0.5846
Epoch 460/1000 | Loss: 0.5846
Epoch 461/1000 | Loss: 0.5845
Epoch 462/1000 | Loss: 0.5845
Epoch 463/1000 | Loss: 0.5844
Epoch 464/1000 | Loss: 0.5843
Epoch 465/1000 | Loss: 0.5843
Epoch 466/1000 | Loss: 0.5842
Epoch 467/1000 | Loss: 0.5842
Epoch 468/1000 | Loss: 0.5841
Epoch 469/1000 | Loss: 0.5840
Epoch 470/1000 | Loss: 0.5840
Epoch 471/1000 | Loss: 0.5839
Epoch 472/1000 | Loss: 0.5839
Epoch 473/1000 | Loss: 0.5838
Epoch 474/1000 | Loss: 0.5837
Epoch 475/1000 | Loss: 0.5837
Epoch 476/1000 | Loss: 0.5836
Epoch 477/1000 | Loss: 0.5836
Epoch 478/1000 | Loss: 0.5835
Epoch 479/1000 | Loss: 0.5835
Epoch 480/1000 | Loss: 0.5834
Epoch 481/1000 | Loss: 0.5833
Epoch 482/1000 | Loss: 0.5833
Epoch 483/1000 | Loss: 0.5832
Epoch 484/1000 | Loss: 0.5832
Epoch 485/1000 | Loss: 0.5831
Epoch 486/1000 | Loss: 0.5831
Epoch 487/1000 | Loss: 0.5830
Epoch 488/1000 | Loss: 0.5829
Epoch 489/1000 | Loss: 0.5829
Epoch 490/1000 | Loss: 0.5828
Epoch 491/1000 | Loss: 0.5828
Epoch 492/1000 | Loss: 0.5827
Epoch 493/1000 | Loss: 0.5827
Epoch 494/1000 | Loss: 0.5826
Epoch 495/1000 | Loss: 0.5825
Epoch 496/1000 | Loss: 0.5825
Epoch 497/1000 | Loss: 0.5824
Epoch 498/1000 | Loss: 0.5824
Epoch 499/1000 | Loss: 0.5823
Epoch 500/1000 | Loss: 0.5823
Epoch 501/1000 | Loss: 0.5822
Epoch 502/1000 | Loss: 0.5822
Epoch 503/1000 | Loss: 0.5821
Epoch 504/1000 | Loss: 0.5820
Epoch 505/1000 | Loss: 0.5820
Epoch 506/1000 | Loss: 0.5819
Epoch 507/1000 | Loss: 0.5819
Epoch 508/1000 | Loss: 0.5818
Epoch 509/1000 | Loss: 0.5818
Epoch 510/1000 | Loss: 0.5817
Epoch 511/1000 | Loss: 0.5817
Epoch 512/1000 | Loss: 0.5816
Epoch 513/1000 | Loss: 0.5815
Epoch 514/1000 | Loss: 0.5815
Epoch 515/1000 | Loss: 0.5814
Epoch 516/1000 | Loss: 0.5814
Epoch 517/1000 | Loss: 0.5813
Epoch 518/1000 | Loss: 0.5813
Epoch 519/1000 | Loss: 0.5812
Epoch 520/1000 | Loss: 0.5812
Epoch 521/1000 | Loss: 0.5811
Epoch 522/1000 | Loss: 0.5811
Epoch 523/1000 | Loss: 0.5810
Epoch 524/1000 | Loss: 0.5810
Epoch 525/1000 | Loss: 0.5809
Epoch 526/1000 | Loss: 0.5808
Epoch 527/1000 | Loss: 0.5808
Epoch 528/1000 | Loss: 0.5807
Epoch 529/1000 | Loss: 0.5807
Epoch 530/1000 | Loss: 0.5806
Epoch 531/1000 | Loss: 0.5806
Epoch 532/1000 | Loss: 0.5805
Epoch 533/1000 | Loss: 0.5805
Epoch 534/1000 | Loss: 0.5804
Epoch 535/1000 | Loss: 0.5804
Epoch 536/1000 | Loss: 0.5803
Epoch 537/1000 | Loss: 0.5803
Epoch 538/1000 | Loss: 0.5802
Epoch 539/1000 | Loss: 0.5802
Epoch 540/1000 | Loss: 0.5801
Epoch 541/1000 | Loss: 0.5801
Epoch 542/1000 | Loss: 0.5800
Epoch 543/1000 | Loss: 0.5799
Epoch 544/1000 | Loss: 0.5799
Epoch 545/1000 | Loss: 0.5798
Epoch 546/1000 | Loss: 0.5798
Epoch 547/1000 | Loss: 0.5797
Epoch 548/1000 | Loss: 0.5797
Epoch 549/1000 | Loss: 0.5796
Epoch 550/1000 | Loss: 0.5796
Epoch 551/1000 | Loss: 0.5795
Epoch 552/1000 | Loss: 0.5795
Epoch 553/1000 | Loss: 0.5794
Epoch 554/1000 | Loss: 0.5794
Epoch 555/1000 | Loss: 0.5793
Epoch 556/1000 | Loss: 0.5793
Epoch 557/1000 | Loss: 0.5792
Epoch 558/1000 | Loss: 0.5792
Epoch 559/1000 | Loss: 0.5791
Epoch 560/1000 | Loss: 0.5791
Epoch 561/1000 | Loss: 0.5790
Epoch 562/1000 | Loss: 0.5790
Epoch 563/1000 | Loss: 0.5789
Epoch 564/1000 | Loss: 0.5789
Epoch 565/1000 | Loss: 0.5788
Epoch 566/1000 | Loss: 0.5788
Epoch 567/1000 | Loss: 0.5787
Epoch 568/1000 | Loss: 0.5787
Epoch 569/1000 | Loss: 0.5786
Epoch 570/1000 | Loss: 0.5786
Epoch 571/1000 | Loss: 0.5785
Epoch 572/1000 | Loss: 0.5785
Epoch 573/1000 | Loss: 0.5784
Epoch 574/1000 | Loss: 0.5784
Epoch 575/1000 | Loss: 0.5783
Epoch 576/1000 | Loss: 0.5783
Epoch 577/1000 | Loss: 0.5782
Epoch 578/1000 | Loss: 0.5782
Epoch 579/1000 | Loss: 0.5781
Epoch 580/1000 | Loss: 0.5781
Epoch 581/1000 | Loss: 0.5780
Epoch 582/1000 | Loss: 0.5780
Epoch 583/1000 | Loss: 0.5779
Epoch 584/1000 | Loss: 0.5779
Epoch 585/1000 | Loss: 0.5778
Epoch 586/1000 | Loss: 0.5778
Epoch 587/1000 | Loss: 0.5777
Epoch 588/1000 | Loss: 0.5777
Epoch 589/1000 | Loss: 0.5776
Epoch 590/1000 | Loss: 0.5776
Epoch 591/1000 | Loss: 0.5775
Epoch 592/1000 | Loss: 0.5775
Epoch 593/1000 | Loss: 0.5774
Epoch 594/1000 | Loss: 0.5774
Epoch 595/1000 | Loss: 0.5773
Epoch 596/1000 | Loss: 0.5773
Epoch 597/1000 | Loss: 0.5772
Epoch 598/1000 | Loss: 0.5772
Epoch 599/1000 | Loss: 0.5771
Epoch 600/1000 | Loss: 0.5771
Epoch 601/1000 | Loss: 0.5770
Epoch 602/1000 | Loss: 0.5770
Epoch 603/1000 | Loss: 0.5769
Epoch 604/1000 | Loss: 0.5769
Epoch 605/1000 | Loss: 0.5768
Epoch 606/1000 | Loss: 0.5768
Epoch 607/1000 | Loss: 0.5767
Epoch 608/1000 | Loss: 0.5767
Epoch 609/1000 | Loss: 0.5767
Epoch 610/1000 | Loss: 0.5766
Epoch 611/1000 | Loss: 0.5766
Epoch 612/1000 | Loss: 0.5765
Epoch 613/1000 | Loss: 0.5765
Epoch 614/1000 | Loss: 0.5764
Epoch 615/1000 | Loss: 0.5764
Epoch 616/1000 | Loss: 0.5763
Epoch 617/1000 | Loss: 0.5763
Epoch 618/1000 | Loss: 0.5762
Epoch 619/1000 | Loss: 0.5762
Epoch 620/1000 | Loss: 0.5761
Epoch 621/1000 | Loss: 0.5761
Epoch 622/1000 | Loss: 0.5760
Epoch 623/1000 | Loss: 0.5760
Epoch 624/1000 | Loss: 0.5759
Epoch 625/1000 | Loss: 0.5759
Epoch 626/1000 | Loss: 0.5758
Epoch 627/1000 | Loss: 0.5758
Epoch 628/1000 | Loss: 0.5757
Epoch 629/1000 | Loss: 0.5757
Epoch 630/1000 | Loss: 0.5757
Epoch 631/1000 | Loss: 0.5756
Epoch 632/1000 | Loss: 0.5756
Epoch 633/1000 | Loss: 0.5755
Epoch 634/1000 | Loss: 0.5755
Epoch 635/1000 | Loss: 0.5754
Epoch 636/1000 | Loss: 0.5754
Epoch 637/1000 | Loss: 0.5753
Epoch 638/1000 | Loss: 0.5753
Epoch 639/1000 | Loss: 0.5752
Epoch 640/1000 | Loss: 0.5752
Epoch 641/1000 | Loss: 0.5751
Epoch 642/1000 | Loss: 0.5751
Epoch 643/1000 | Loss: 0.5750
Epoch 644/1000 | Loss: 0.5750
Epoch 645/1000 | Loss: 0.5750
Epoch 646/1000 | Loss: 0.5749
Epoch 647/1000 | Loss: 0.5749
Epoch 648/1000 | Loss: 0.5748
Epoch 649/1000 | Loss: 0.5748
Epoch 650/1000 | Loss: 0.5747
Epoch 651/1000 | Loss: 0.5747
Epoch 652/1000 | Loss: 0.5746
Epoch 653/1000 | Loss: 0.5746
Epoch 654/1000 | Loss: 0.5745
Epoch 655/1000 | Loss: 0.5745
Epoch 656/1000 | Loss: 0.5744
Epoch 657/1000 | Loss: 0.5744
Epoch 658/1000 | Loss: 0.5744
Epoch 659/1000 | Loss: 0.5743
Epoch 660/1000 | Loss: 0.5743
Epoch 661/1000 | Loss: 0.5742
Epoch 662/1000 | Loss: 0.5742
Epoch 663/1000 | Loss: 0.5741
Epoch 664/1000 | Loss: 0.5741
Epoch 665/1000 | Loss: 0.5740
Epoch 666/1000 | Loss: 0.5740
Epoch 667/1000 | Loss: 0.5739
Epoch 668/1000 | Loss: 0.5739
Epoch 669/1000 | Loss: 0.5739
Epoch 670/1000 | Loss: 0.5738
Epoch 671/1000 | Loss: 0.5738
Epoch 672/1000 | Loss: 0.5737
Epoch 673/1000 | Loss: 0.5737
Epoch 674/1000 | Loss: 0.5736
Epoch 675/1000 | Loss: 0.5736
Epoch 676/1000 | Loss: 0.5735
Epoch 677/1000 | Loss: 0.5735
Epoch 678/1000 | Loss: 0.5734
Epoch 679/1000 | Loss: 0.5734
Epoch 680/1000 | Loss: 0.5734
Epoch 681/1000 | Loss: 0.5733
Epoch 682/1000 | Loss: 0.5733
Epoch 683/1000 | Loss: 0.5732
Epoch 684/1000 | Loss: 0.5732
Epoch 685/1000 | Loss: 0.5731
Epoch 686/1000 | Loss: 0.5731
Epoch 687/1000 | Loss: 0.5730
Epoch 688/1000 | Loss: 0.5730
Epoch 689/1000 | Loss: 0.5730
Epoch 690/1000 | Loss: 0.5729
Epoch 691/1000 | Loss: 0.5729
Epoch 692/1000 | Loss: 0.5728
Epoch 693/1000 | Loss: 0.5728
Epoch 694/1000 | Loss: 0.5727
Epoch 695/1000 | Loss: 0.5727
Epoch 696/1000 | Loss: 0.5726
Epoch 697/1000 | Loss: 0.5726
Epoch 698/1000 | Loss: 0.5725
Epoch 699/1000 | Loss: 0.5725
Epoch 700/1000 | Loss: 0.5725
Epoch 701/1000 | Loss: 0.5724
Epoch 702/1000 | Loss: 0.5724
Epoch 703/1000 | Loss: 0.5723
Epoch 704/1000 | Loss: 0.5723
Epoch 705/1000 | Loss: 0.5722
Epoch 706/1000 | Loss: 0.5722
Epoch 707/1000 | Loss: 0.5722
Epoch 708/1000 | Loss: 0.5721
Epoch 709/1000 | Loss: 0.5721
Epoch 710/1000 | Loss: 0.5720
Epoch 711/1000 | Loss: 0.5720
Epoch 712/1000 | Loss: 0.5719
Epoch 713/1000 | Loss: 0.5719
Epoch 714/1000 | Loss: 0.5718
Epoch 715/1000 | Loss: 0.5718
Epoch 716/1000 | Loss: 0.5718
Epoch 717/1000 | Loss: 0.5717
Epoch 718/1000 | Loss: 0.5717
Epoch 719/1000 | Loss: 0.5716
Epoch 720/1000 | Loss: 0.5716
Epoch 721/1000 | Loss: 0.5715
Epoch 722/1000 | Loss: 0.5715
Epoch 723/1000 | Loss: 0.5715
Epoch 724/1000 | Loss: 0.5714
Epoch 725/1000 | Loss: 0.5714
Epoch 726/1000 | Loss: 0.5713
Epoch 727/1000 | Loss: 0.5713
Epoch 728/1000 | Loss: 0.5712
Epoch 729/1000 | Loss: 0.5712
Epoch 730/1000 | Loss: 0.5711
Epoch 731/1000 | Loss: 0.5711
Epoch 732/1000 | Loss: 0.5711
Epoch 733/1000 | Loss: 0.5710
Epoch 734/1000 | Loss: 0.5710
Epoch 735/1000 | Loss: 0.5709
Epoch 736/1000 | Loss: 0.5709
Epoch 737/1000 | Loss: 0.5708
Epoch 738/1000 | Loss: 0.5708
Epoch 739/1000 | Loss: 0.5708
Epoch 740/1000 | Loss: 0.5707
Epoch 741/1000 | Loss: 0.5707
Epoch 742/1000 | Loss: 0.5706
Epoch 743/1000 | Loss: 0.5706
Epoch 744/1000 | Loss: 0.5705
Epoch 745/1000 | Loss: 0.5705
Epoch 746/1000 | Loss: 0.5705
Epoch 747/1000 | Loss: 0.5704
Epoch 748/1000 | Loss: 0.5704
Epoch 749/1000 | Loss: 0.5703
Epoch 750/1000 | Loss: 0.5703
Epoch 751/1000 | Loss: 0.5702
Epoch 752/1000 | Loss: 0.5702
Epoch 753/1000 | Loss: 0.5702
Epoch 754/1000 | Loss: 0.5701
Epoch 755/1000 | Loss: 0.5701
Epoch 756/1000 | Loss: 0.5700
Epoch 757/1000 | Loss: 0.5700
Epoch 758/1000 | Loss: 0.5700
Epoch 759/1000 | Loss: 0.5699
Epoch 760/1000 | Loss: 0.5699
Epoch 761/1000 | Loss: 0.5698
Epoch 762/1000 | Loss: 0.5698
Epoch 763/1000 | Loss: 0.5697
Epoch 764/1000 | Loss: 0.5697
Epoch 765/1000 | Loss: 0.5697
Epoch 766/1000 | Loss: 0.5696
Epoch 767/1000 | Loss: 0.5696
Epoch 768/1000 | Loss: 0.5695
Epoch 769/1000 | Loss: 0.5695
Epoch 770/1000 | Loss: 0.5694
Epoch 771/1000 | Loss: 0.5694
Epoch 772/1000 | Loss: 0.5694
Epoch 773/1000 | Loss: 0.5693
Epoch 774/1000 | Loss: 0.5693
Epoch 775/1000 | Loss: 0.5692
Epoch 776/1000 | Loss: 0.5692
Epoch 777/1000 | Loss: 0.5692
Epoch 778/1000 | Loss: 0.5691
Epoch 779/1000 | Loss: 0.5691
Epoch 780/1000 | Loss: 0.5690
Epoch 781/1000 | Loss: 0.5690
Epoch 782/1000 | Loss: 0.5689
Epoch 783/1000 | Loss: 0.5689
Epoch 784/1000 | Loss: 0.5689
Epoch 785/1000 | Loss: 0.5688
Epoch 786/1000 | Loss: 0.5688
Epoch 787/1000 | Loss: 0.5687
Epoch 788/1000 | Loss: 0.5687
Epoch 789/1000 | Loss: 0.5687
Epoch 790/1000 | Loss: 0.5686
Epoch 791/1000 | Loss: 0.5686
Epoch 792/1000 | Loss: 0.5685
Epoch 793/1000 | Loss: 0.5685
Epoch 794/1000 | Loss: 0.5684
Epoch 795/1000 | Loss: 0.5684
Epoch 796/1000 | Loss: 0.5684
Epoch 797/1000 | Loss: 0.5683
Epoch 798/1000 | Loss: 0.5683
Epoch 799/1000 | Loss: 0.5682
Epoch 800/1000 | Loss: 0.5682
Epoch 801/1000 | Loss: 0.5682
Epoch 802/1000 | Loss: 0.5681
Epoch 803/1000 | Loss: 0.5681
Epoch 804/1000 | Loss: 0.5680
Epoch 805/1000 | Loss: 0.5680
Epoch 806/1000 | Loss: 0.5680
Epoch 807/1000 | Loss: 0.5679
Epoch 808/1000 | Loss: 0.5679
Epoch 809/1000 | Loss: 0.5678
Epoch 810/1000 | Loss: 0.5678
Epoch 811/1000 | Loss: 0.5678
Epoch 812/1000 | Loss: 0.5677
Epoch 813/1000 | Loss: 0.5677
Epoch 814/1000 | Loss: 0.5676
Epoch 815/1000 | Loss: 0.5676
Epoch 816/1000 | Loss: 0.5675
Epoch 817/1000 | Loss: 0.5675
Epoch 818/1000 | Loss: 0.5675
Epoch 819/1000 | Loss: 0.5674
Epoch 820/1000 | Loss: 0.5674
Epoch 821/1000 | Loss: 0.5673
Epoch 822/1000 | Loss: 0.5673
Epoch 823/1000 | Loss: 0.5673
Epoch 824/1000 | Loss: 0.5672
Epoch 825/1000 | Loss: 0.5672
Epoch 826/1000 | Loss: 0.5671
Epoch 827/1000 | Loss: 0.5671
Epoch 828/1000 | Loss: 0.5671
Epoch 829/1000 | Loss: 0.5670
Epoch 830/1000 | Loss: 0.5670
Epoch 831/1000 | Loss: 0.5669
Epoch 832/1000 | Loss: 0.5669
Epoch 833/1000 | Loss: 0.5669
Epoch 834/1000 | Loss: 0.5668
Epoch 835/1000 | Loss: 0.5668
Epoch 836/1000 | Loss: 0.5667
Epoch 837/1000 | Loss: 0.5667
Epoch 838/1000 | Loss: 0.5667
Epoch 839/1000 | Loss: 0.5666
Epoch 840/1000 | Loss: 0.5666
Epoch 841/1000 | Loss: 0.5665
Epoch 842/1000 | Loss: 0.5665
Epoch 843/1000 | Loss: 0.5665
Epoch 844/1000 | Loss: 0.5664
Epoch 845/1000 | Loss: 0.5664
Epoch 846/1000 | Loss: 0.5663
Epoch 847/1000 | Loss: 0.5663
Epoch 848/1000 | Loss: 0.5663
Epoch 849/1000 | Loss: 0.5662
Epoch 850/1000 | Loss: 0.5662
Epoch 851/1000 | Loss: 0.5661
Epoch 852/1000 | Loss: 0.5661
Epoch 853/1000 | Loss: 0.5661
Epoch 854/1000 | Loss: 0.5660
Epoch 855/1000 | Loss: 0.5660
Epoch 856/1000 | Loss: 0.5659
Epoch 857/1000 | Loss: 0.5659
Epoch 858/1000 | Loss: 0.5659
Epoch 859/1000 | Loss: 0.5658
Epoch 860/1000 | Loss: 0.5658
Epoch 861/1000 | Loss: 0.5657
Epoch 862/1000 | Loss: 0.5657
Epoch 863/1000 | Loss: 0.5657
Epoch 864/1000 | Loss: 0.5656
Epoch 865/1000 | Loss: 0.5656
Epoch 866/1000 | Loss: 0.5655
Epoch 867/1000 | Loss: 0.5655
Epoch 868/1000 | Loss: 0.5655
Epoch 869/1000 | Loss: 0.5654
Epoch 870/1000 | Loss: 0.5654
Epoch 871/1000 | Loss: 0.5654
Epoch 872/1000 | Loss: 0.5653
Epoch 873/1000 | Loss: 0.5653
Epoch 874/1000 | Loss: 0.5652
Epoch 875/1000 | Loss: 0.5652
Epoch 876/1000 | Loss: 0.5652
Epoch 877/1000 | Loss: 0.5651
Epoch 878/1000 | Loss: 0.5651
Epoch 879/1000 | Loss: 0.5650
Epoch 880/1000 | Loss: 0.5650
Epoch 881/1000 | Loss: 0.5650
Epoch 882/1000 | Loss: 0.5649
Epoch 883/1000 | Loss: 0.5649
Epoch 884/1000 | Loss: 0.5648
Epoch 885/1000 | Loss: 0.5648
Epoch 886/1000 | Loss: 0.5648
Epoch 887/1000 | Loss: 0.5647
Epoch 888/1000 | Loss: 0.5647
Epoch 889/1000 | Loss: 0.5646
Epoch 890/1000 | Loss: 0.5646
Epoch 891/1000 | Loss: 0.5646
Epoch 892/1000 | Loss: 0.5645
Epoch 893/1000 | Loss: 0.5645
Epoch 894/1000 | Loss: 0.5645
Epoch 895/1000 | Loss: 0.5644
Epoch 896/1000 | Loss: 0.5644
Epoch 897/1000 | Loss: 0.5643
Epoch 898/1000 | Loss: 0.5643
Epoch 899/1000 | Loss: 0.5643
Epoch 900/1000 | Loss: 0.5642
Epoch 901/1000 | Loss: 0.5642
Epoch 902/1000 | Loss: 0.5641
Epoch 903/1000 | Loss: 0.5641
Epoch 904/1000 | Loss: 0.5641
Epoch 905/1000 | Loss: 0.5640
Epoch 906/1000 | Loss: 0.5640
Epoch 907/1000 | Loss: 0.5640
Epoch 908/1000 | Loss: 0.5639
Epoch 909/1000 | Loss: 0.5639
Epoch 910/1000 | Loss: 0.5638
Epoch 911/1000 | Loss: 0.5638
Epoch 912/1000 | Loss: 0.5638
Epoch 913/1000 | Loss: 0.5637
Epoch 914/1000 | Loss: 0.5637
Epoch 915/1000 | Loss: 0.5636
Epoch 916/1000 | Loss: 0.5636
Epoch 917/1000 | Loss: 0.5636
Epoch 918/1000 | Loss: 0.5635
Epoch 919/1000 | Loss: 0.5635
Epoch 920/1000 | Loss: 0.5635
Epoch 921/1000 | Loss: 0.5634
Epoch 922/1000 | Loss: 0.5634
Epoch 923/1000 | Loss: 0.5633
Epoch 924/1000 | Loss: 0.5633
Epoch 925/1000 | Loss: 0.5633
Epoch 926/1000 | Loss: 0.5632
Epoch 927/1000 | Loss: 0.5632
Epoch 928/1000 | Loss: 0.5632
Epoch 929/1000 | Loss: 0.5631
Epoch 930/1000 | Loss: 0.5631
Epoch 931/1000 | Loss: 0.5630
Epoch 932/1000 | Loss: 0.5630
Epoch 933/1000 | Loss: 0.5630
Epoch 934/1000 | Loss: 0.5629
Epoch 935/1000 | Loss: 0.5629
Epoch 936/1000 | Loss: 0.5629
Epoch 937/1000 | Loss: 0.5628
Epoch 938/1000 | Loss: 0.5628
Epoch 939/1000 | Loss: 0.5627
Epoch 940/1000 | Loss: 0.5627
Epoch 941/1000 | Loss: 0.5627
Epoch 942/1000 | Loss: 0.5626
Epoch 943/1000 | Loss: 0.5626
Epoch 944/1000 | Loss: 0.5626
Epoch 945/1000 | Loss: 0.5625
Epoch 946/1000 | Loss: 0.5625
Epoch 947/1000 | Loss: 0.5624
Epoch 948/1000 | Loss: 0.5624
Epoch 949/1000 | Loss: 0.5624
Epoch 950/1000 | Loss: 0.5623
Epoch 951/1000 | Loss: 0.5623
Epoch 952/1000 | Loss: 0.5623
Epoch 953/1000 | Loss: 0.5622
Epoch 954/1000 | Loss: 0.5622
Epoch 955/1000 | Loss: 0.5621
Epoch 956/1000 | Loss: 0.5621
Epoch 957/1000 | Loss: 0.5621
Epoch 958/1000 | Loss: 0.5620
Epoch 959/1000 | Loss: 0.5620
Epoch 960/1000 | Loss: 0.5620
Epoch 961/1000 | Loss: 0.5619
Epoch 962/1000 | Loss: 0.5619
Epoch 963/1000 | Loss: 0.5618
Epoch 964/1000 | Loss: 0.5618
Epoch 965/1000 | Loss: 0.5618
Epoch 966/1000 | Loss: 0.5617
Epoch 967/1000 | Loss: 0.5617
Epoch 968/1000 | Loss: 0.5617
Epoch 969/1000 | Loss: 0.5616
Epoch 970/1000 | Loss: 0.5616
Epoch 971/1000 | Loss: 0.5615
Epoch 972/1000 | Loss: 0.5615
Epoch 973/1000 | Loss: 0.5615
Epoch 974/1000 | Loss: 0.5614
Epoch 975/1000 | Loss: 0.5614
Epoch 976/1000 | Loss: 0.5614
Epoch 977/1000 | Loss: 0.5613
Epoch 978/1000 | Loss: 0.5613
Epoch 979/1000 | Loss: 0.5613
Epoch 980/1000 | Loss: 0.5612
Epoch 981/1000 | Loss: 0.5612
Epoch 982/1000 | Loss: 0.5611
Epoch 983/1000 | Loss: 0.5611
Epoch 984/1000 | Loss: 0.5611
Epoch 985/1000 | Loss: 0.5610
Epoch 986/1000 | Loss: 0.5610
Epoch 987/1000 | Loss: 0.5610
Epoch 988/1000 | Loss: 0.5609
Epoch 989/1000 | Loss: 0.5609
Epoch 990/1000 | Loss: 0.5609
Epoch 991/1000 | Loss: 0.5608
Epoch 992/1000 | Loss: 0.5608
Epoch 993/1000 | Loss: 0.5607
Epoch 994/1000 | Loss: 0.5607
Epoch 995/1000 | Loss: 0.5607
Epoch 996/1000 | Loss: 0.5606
Epoch 997/1000 | Loss: 0.5606
Epoch 998/1000 | Loss: 0.5606
Epoch 999/1000 | Loss: 0.5605
Epoch 1000/1000 | Loss: 0.5605

Let's predict the hours need to score above 50%
==================================================
Prediction after 1 hours of training: 0.4926 | Above 50%: False
Accuracy :  46.9038208168643 %
