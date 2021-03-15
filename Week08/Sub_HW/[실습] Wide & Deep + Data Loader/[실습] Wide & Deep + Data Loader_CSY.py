import torch
from torch import nn, optim, from_numpy
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
from torch import sigmoid
from torch import tensor
from sklearn.model_selection import train_test_split

xy = np.loadtxt('/content/data/diabetes.csv',delimiter=',', dtype=np.float32)
x_data = from_numpy(xy[:, 0:-1])
y_data = from_numpy(xy[:, [-1]])
x_train, x_test, y_train, y_test = train_test_split(x_data,y_data,test_size=0.077,shuffle=False)

class DiabetesDataset(Dataset):
    # 모델의 생성자 정의
    def __init__(self):
        self.len = x_train.shape[0]
        self.x_train = x_train
        self.y_train = y_train

    # 모델 객체와 index를 받아 해당 x_data와 y_data를 반환
    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index]

    # 모델 객체를 받아 데이터 길이 반환
    def __len__(self):
        return self.len

dataset = DiabetesDataset()
train_loader = DataLoader(dataset=dataset,batch_size=25,shuffle=True,num_workers=0)

correct=0
answer=0

# torch.nn.Module을 상속받는 파이썬 클래스 정의
# 모델의 생성자 정의
class Model(nn.Module):
    def __init__(self):
        # super() 함수를 통해 nn.Module의 생성자를 호출함
        super(Model, self).__init__()
        self.l1 = nn.Linear(8, 6)
        self.l2 = nn.Linear(6, 4)
        self.l3 = nn.Linear(4, 1)

        self.sigmoid = nn.Sigmoid()

    # 모델 객체와 학습 데이터인 x를 받아 forward 연산
    def forward(self, x):
        out1 = self.sigmoid(self.l1(x))
        out2 = self.sigmoid(self.l2(out1))
        y_pred = self.sigmoid(self.l3(out2))
        return y_pred

# Model 클래스 변수 model 선언
model = Model()

# 손실함수 정의
# SGD(확률적 경사 하강법)는 경사 하강법의 설정
criterion = nn.BCELoss(reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=0.25)

for epoch in range(500):
  for i, data in enumerate(train_loader, 0):
    inputs, labels = data
    inputs, labels = tensor(inputs), tensor(labels)

    y_pred = model(inputs)
    loss = criterion(y_pred, labels)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  print(f'Epoch {epoch + 1}/500 | Loss: {loss.item():.4f}')

y_pred = model(x_test)

# 정확도 계산, 맞춘 개수를 셈
for i in range(len(y_test)):
    if y_pred[i] > 0.5:
        answer=1
    else:
        answer=0
    if answer == y_test[i]:
        correct+=1

# 정확도 출력
print(f'accuracy: {correct/len(y_test)*100}')

'''
Epoch 1/500 | Loss: 0.6534
Epoch 2/500 | Loss: 0.5271
Epoch 3/500 | Loss: 0.6284
Epoch 4/500 | Loss: 0.6760
Epoch 5/500 | Loss: 0.7374
Epoch 6/500 | Loss: 0.7088
Epoch 7/500 | Loss: 0.6545
Epoch 8/500 | Loss: 0.6267
Epoch 9/500 | Loss: 0.5789
Epoch 10/500 | Loss: 0.6546
Epoch 11/500 | Loss: 0.5992
Epoch 12/500 | Loss: 0.6831
Epoch 13/500 | Loss: 0.6864
Epoch 14/500 | Loss: 0.6552
Epoch 15/500 | Loss: 0.6308
Epoch 16/500 | Loss: 0.7187
Epoch 17/500 | Loss: 0.5503
Epoch 18/500 | Loss: 0.6529
Epoch 19/500 | Loss: 0.5339
Epoch 20/500 | Loss: 0.6563
Epoch 21/500 | Loss: 0.5998
Epoch 22/500 | Loss: 0.6273
Epoch 23/500 | Loss: 0.5308
Epoch 24/500 | Loss: 0.5760
Epoch 25/500 | Loss: 0.5816
Epoch 26/500 | Loss: 0.6054
Epoch 27/500 | Loss: 0.6259
Epoch 28/500 | Loss: 0.7370
Epoch 29/500 | Loss: 0.6817
Epoch 30/500 | Loss: 0.5762
Epoch 31/500 | Loss: 0.5675
Epoch 32/500 | Loss: 0.6521
Epoch 33/500 | Loss: 0.6264
Epoch 34/500 | Loss: 0.6543
Epoch 35/500 | Loss: 0.6515
Epoch 36/500 | Loss: 0.5451
Epoch 37/500 | Loss: 0.5802
Epoch 38/500 | Loss: 0.7128
Epoch 39/500 | Loss: 0.7035
Epoch 40/500 | Loss: 0.5299
Epoch 41/500 | Loss: 0.6518
Epoch 42/500 | Loss: 0.6250
Epoch 43/500 | Loss: 0.5990
Epoch 44/500 | Loss: 0.6977
Epoch 45/500 | Loss: 0.6256
Epoch 46/500 | Loss: 0.6510
Epoch 47/500 | Loss: 0.5522
Epoch 48/500 | Loss: 0.6774
Epoch 49/500 | Loss: 0.5239
Epoch 50/500 | Loss: 0.5868
Epoch 51/500 | Loss: 0.5493
Epoch 52/500 | Loss: 0.7114
Epoch 53/500 | Loss: 0.5966
Epoch 54/500 | Loss: 0.5800
Epoch 55/500 | Loss: 0.6466
Epoch 56/500 | Loss: 0.5697
Epoch 57/500 | Loss: 0.5683
Epoch 58/500 | Loss: 0.6964
Epoch 59/500 | Loss: 0.5813
Epoch 60/500 | Loss: 0.6451
Epoch 61/500 | Loss: 0.6588
Epoch 62/500 | Loss: 0.6168
Epoch 63/500 | Loss: 0.5837
Epoch 64/500 | Loss: 0.6815
Epoch 65/500 | Loss: 0.5816
Epoch 66/500 | Loss: 0.5578
Epoch 67/500 | Loss: 0.5091
Epoch 68/500 | Loss: 0.5673
Epoch 69/500 | Loss: 0.4502
Epoch 70/500 | Loss: 0.5297
Epoch 71/500 | Loss: 0.5310
Epoch 72/500 | Loss: 0.5450
Epoch 73/500 | Loss: 0.5343
Epoch 74/500 | Loss: 0.4291
Epoch 75/500 | Loss: 0.5749
Epoch 76/500 | Loss: 0.4316
Epoch 77/500 | Loss: 0.6217
Epoch 78/500 | Loss: 0.4609
Epoch 79/500 | Loss: 0.4514
Epoch 80/500 | Loss: 0.6030
Epoch 81/500 | Loss: 0.4209
Epoch 82/500 | Loss: 0.5141
Epoch 83/500 | Loss: 0.4619
Epoch 84/500 | Loss: 0.4958
Epoch 85/500 | Loss: 0.5398
Epoch 86/500 | Loss: 0.5330
Epoch 87/500 | Loss: 0.5039
Epoch 88/500 | Loss: 0.4560
Epoch 89/500 | Loss: 0.5828
Epoch 90/500 | Loss: 0.4474
Epoch 91/500 | Loss: 0.4742
Epoch 92/500 | Loss: 0.5655
Epoch 93/500 | Loss: 0.3756
Epoch 94/500 | Loss: 0.4131
Epoch 95/500 | Loss: 0.5187
Epoch 96/500 | Loss: 0.5165
Epoch 97/500 | Loss: 0.3947
Epoch 98/500 | Loss: 0.3482
Epoch 99/500 | Loss: 0.5739
Epoch 100/500 | Loss: 0.3400
Epoch 101/500 | Loss: 0.5156
Epoch 102/500 | Loss: 0.4742
Epoch 103/500 | Loss: 0.3478
Epoch 104/500 | Loss: 0.3745
Epoch 105/500 | Loss: 0.4123
Epoch 106/500 | Loss: 0.4654
Epoch 107/500 | Loss: 0.4722
Epoch 108/500 | Loss: 0.5389
Epoch 109/500 | Loss: 0.4894
Epoch 110/500 | Loss: 0.6260
Epoch 111/500 | Loss: 0.3814
Epoch 112/500 | Loss: 0.6120
Epoch 113/500 | Loss: 0.4777
Epoch 114/500 | Loss: 0.4318
Epoch 115/500 | Loss: 0.5877
Epoch 116/500 | Loss: 0.4383
Epoch 117/500 | Loss: 0.5584
Epoch 118/500 | Loss: 0.4098
Epoch 119/500 | Loss: 0.3513
Epoch 120/500 | Loss: 0.5019
Epoch 121/500 | Loss: 0.4266
Epoch 122/500 | Loss: 0.5167
Epoch 123/500 | Loss: 0.6345
Epoch 124/500 | Loss: 0.3187
Epoch 125/500 | Loss: 0.5025
Epoch 126/500 | Loss: 0.5117
Epoch 127/500 | Loss: 0.5460
Epoch 128/500 | Loss: 0.6653
Epoch 129/500 | Loss: 0.6120
Epoch 130/500 | Loss: 0.4376
Epoch 131/500 | Loss: 0.4865
Epoch 132/500 | Loss: 0.4881
Epoch 133/500 | Loss: 0.4535
Epoch 134/500 | Loss: 0.5115
Epoch 135/500 | Loss: 0.4624
Epoch 136/500 | Loss: 0.4523
Epoch 137/500 | Loss: 0.3904
Epoch 138/500 | Loss: 0.4124
Epoch 139/500 | Loss: 0.4993
Epoch 140/500 | Loss: 0.4304
Epoch 141/500 | Loss: 0.3800
Epoch 142/500 | Loss: 0.4020
Epoch 143/500 | Loss: 0.5989
Epoch 144/500 | Loss: 0.5881
Epoch 145/500 | Loss: 0.4329
Epoch 146/500 | Loss: 0.6701
Epoch 147/500 | Loss: 0.5139
Epoch 148/500 | Loss: 0.3728
Epoch 149/500 | Loss: 0.5400
Epoch 150/500 | Loss: 0.4851
Epoch 151/500 | Loss: 0.4555
Epoch 152/500 | Loss: 0.6009
Epoch 153/500 | Loss: 0.4368
Epoch 154/500 | Loss: 0.4765
Epoch 155/500 | Loss: 0.4073
Epoch 156/500 | Loss: 0.4369
Epoch 157/500 | Loss: 0.4597
Epoch 158/500 | Loss: 0.4117
Epoch 159/500 | Loss: 0.5351
Epoch 160/500 | Loss: 0.3814
Epoch 161/500 | Loss: 0.4390
Epoch 162/500 | Loss: 0.5170
Epoch 163/500 | Loss: 0.4001
Epoch 164/500 | Loss: 0.3942
Epoch 165/500 | Loss: 0.4426
Epoch 166/500 | Loss: 0.5107
Epoch 167/500 | Loss: 0.4340
Epoch 168/500 | Loss: 0.3942
Epoch 169/500 | Loss: 0.4811
Epoch 170/500 | Loss: 0.5886
Epoch 171/500 | Loss: 0.4135
Epoch 172/500 | Loss: 0.4568
Epoch 173/500 | Loss: 0.3209
Epoch 174/500 | Loss: 0.5293
Epoch 175/500 | Loss: 0.6856
Epoch 176/500 | Loss: 0.3975
Epoch 177/500 | Loss: 0.5910
Epoch 178/500 | Loss: 0.4750
Epoch 179/500 | Loss: 0.5947
Epoch 180/500 | Loss: 0.5292
Epoch 181/500 | Loss: 0.5867
Epoch 182/500 | Loss: 0.4694
Epoch 183/500 | Loss: 0.4047
Epoch 184/500 | Loss: 0.6779
Epoch 185/500 | Loss: 0.4028
Epoch 186/500 | Loss: 0.4057
Epoch 187/500 | Loss: 0.3596
Epoch 188/500 | Loss: 0.5635
Epoch 189/500 | Loss: 0.5946
Epoch 190/500 | Loss: 0.6460
Epoch 191/500 | Loss: 0.4083
Epoch 192/500 | Loss: 0.3590
Epoch 193/500 | Loss: 0.5621
Epoch 194/500 | Loss: 0.3130
Epoch 195/500 | Loss: 0.5396
Epoch 196/500 | Loss: 0.3910
Epoch 197/500 | Loss: 0.4829
Epoch 198/500 | Loss: 0.4303
Epoch 199/500 | Loss: 0.4210
Epoch 200/500 | Loss: 0.5590
Epoch 201/500 | Loss: 0.4358
Epoch 202/500 | Loss: 0.2736
Epoch 203/500 | Loss: 0.2996
Epoch 204/500 | Loss: 0.4770
Epoch 205/500 | Loss: 0.3808
Epoch 206/500 | Loss: 0.4771
Epoch 207/500 | Loss: 0.3848
Epoch 208/500 | Loss: 0.4922
Epoch 209/500 | Loss: 0.3902
Epoch 210/500 | Loss: 0.4244
Epoch 211/500 | Loss: 0.3446
Epoch 212/500 | Loss: 0.4100
Epoch 213/500 | Loss: 0.3701
Epoch 214/500 | Loss: 0.5047
Epoch 215/500 | Loss: 0.4770
Epoch 216/500 | Loss: 0.3790
Epoch 217/500 | Loss: 0.5499
Epoch 218/500 | Loss: 0.7037
Epoch 219/500 | Loss: 0.4138
Epoch 220/500 | Loss: 0.3595
Epoch 221/500 | Loss: 0.3196
Epoch 222/500 | Loss: 0.6742
Epoch 223/500 | Loss: 0.3874
Epoch 224/500 | Loss: 0.4026
Epoch 225/500 | Loss: 0.4622
Epoch 226/500 | Loss: 0.5789
Epoch 227/500 | Loss: 0.4663
Epoch 228/500 | Loss: 0.4881
Epoch 229/500 | Loss: 0.3521
Epoch 230/500 | Loss: 0.5244
Epoch 231/500 | Loss: 0.6519
Epoch 232/500 | Loss: 0.2949
Epoch 233/500 | Loss: 0.4084
Epoch 234/500 | Loss: 0.5504
Epoch 235/500 | Loss: 0.4193
Epoch 236/500 | Loss: 0.6695
Epoch 237/500 | Loss: 0.3509
Epoch 238/500 | Loss: 0.3747
Epoch 239/500 | Loss: 0.5893
Epoch 240/500 | Loss: 0.3310
Epoch 241/500 | Loss: 0.3952
Epoch 242/500 | Loss: 0.4184
Epoch 243/500 | Loss: 0.3961
Epoch 244/500 | Loss: 0.4020
Epoch 245/500 | Loss: 0.6585
Epoch 246/500 | Loss: 0.5145
Epoch 247/500 | Loss: 0.3673
Epoch 248/500 | Loss: 0.5662
Epoch 249/500 | Loss: 0.6371
Epoch 250/500 | Loss: 0.5187
Epoch 251/500 | Loss: 0.5344
Epoch 252/500 | Loss: 0.3932
Epoch 253/500 | Loss: 0.4565
Epoch 254/500 | Loss: 0.4729
Epoch 255/500 | Loss: 0.5992
Epoch 256/500 | Loss: 0.3754
Epoch 257/500 | Loss: 0.5468
Epoch 258/500 | Loss: 0.4976
Epoch 259/500 | Loss: 0.3898
Epoch 260/500 | Loss: 0.2633
Epoch 261/500 | Loss: 0.4330
Epoch 262/500 | Loss: 0.4368
Epoch 263/500 | Loss: 0.4459
Epoch 264/500 | Loss: 0.4527
Epoch 265/500 | Loss: 0.4015
Epoch 266/500 | Loss: 0.5377
Epoch 267/500 | Loss: 0.4517
Epoch 268/500 | Loss: 0.5301
Epoch 269/500 | Loss: 0.4284
Epoch 270/500 | Loss: 0.3580
Epoch 271/500 | Loss: 0.5076
Epoch 272/500 | Loss: 0.4427
Epoch 273/500 | Loss: 0.4538
Epoch 274/500 | Loss: 0.5009
Epoch 275/500 | Loss: 0.6941
Epoch 276/500 | Loss: 0.3489
Epoch 277/500 | Loss: 0.6055
Epoch 278/500 | Loss: 0.4420
Epoch 279/500 | Loss: 0.4679
Epoch 280/500 | Loss: 0.2809
Epoch 281/500 | Loss: 0.4811
Epoch 282/500 | Loss: 0.6809
Epoch 283/500 | Loss: 0.5115
Epoch 284/500 | Loss: 0.4898
Epoch 285/500 | Loss: 0.3341
Epoch 286/500 | Loss: 0.4910
Epoch 287/500 | Loss: 0.2770
Epoch 288/500 | Loss: 0.5961
Epoch 289/500 | Loss: 0.3808
Epoch 290/500 | Loss: 0.3708
Epoch 291/500 | Loss: 0.3109
Epoch 292/500 | Loss: 0.3361
Epoch 293/500 | Loss: 0.5647
Epoch 294/500 | Loss: 0.4260
Epoch 295/500 | Loss: 0.3164
Epoch 296/500 | Loss: 0.3900
Epoch 297/500 | Loss: 0.5995
Epoch 298/500 | Loss: 0.4377
Epoch 299/500 | Loss: 0.3346
Epoch 300/500 | Loss: 0.3942
Epoch 301/500 | Loss: 0.5015
Epoch 302/500 | Loss: 0.5181
Epoch 303/500 | Loss: 0.6320
Epoch 304/500 | Loss: 0.5942
Epoch 305/500 | Loss: 0.4221
Epoch 306/500 | Loss: 0.5033
Epoch 307/500 | Loss: 0.3824
Epoch 308/500 | Loss: 0.4559
Epoch 309/500 | Loss: 0.3428
Epoch 310/500 | Loss: 0.4859
Epoch 311/500 | Loss: 0.4512
Epoch 312/500 | Loss: 0.5008
Epoch 313/500 | Loss: 0.4405
Epoch 314/500 | Loss: 0.3755
Epoch 315/500 | Loss: 0.4314
Epoch 316/500 | Loss: 0.3929
Epoch 317/500 | Loss: 0.5414
Epoch 318/500 | Loss: 0.4588
Epoch 319/500 | Loss: 0.3775
Epoch 320/500 | Loss: 0.5421
Epoch 321/500 | Loss: 0.3810
Epoch 322/500 | Loss: 0.5409
Epoch 323/500 | Loss: 0.5204
Epoch 324/500 | Loss: 0.3713
Epoch 325/500 | Loss: 0.3789
Epoch 326/500 | Loss: 0.4860
Epoch 327/500 | Loss: 0.4708
Epoch 328/500 | Loss: 0.4072
Epoch 329/500 | Loss: 0.4445
Epoch 330/500 | Loss: 0.5285
Epoch 331/500 | Loss: 0.4674
Epoch 332/500 | Loss: 0.6501
Epoch 333/500 | Loss: 0.5395
Epoch 334/500 | Loss: 0.5171
Epoch 335/500 | Loss: 0.5647
Epoch 336/500 | Loss: 0.5207
Epoch 337/500 | Loss: 0.4403
Epoch 338/500 | Loss: 0.4195
Epoch 339/500 | Loss: 0.5187
Epoch 340/500 | Loss: 0.2621
Epoch 341/500 | Loss: 0.4180
Epoch 342/500 | Loss: 0.4040
Epoch 343/500 | Loss: 0.5133
Epoch 344/500 | Loss: 0.3894
Epoch 345/500 | Loss: 0.4911
Epoch 346/500 | Loss: 0.3863
Epoch 347/500 | Loss: 0.3973
Epoch 348/500 | Loss: 0.6357
Epoch 349/500 | Loss: 0.3910
Epoch 350/500 | Loss: 0.4707
Epoch 351/500 | Loss: 0.4045
Epoch 352/500 | Loss: 0.7568
Epoch 353/500 | Loss: 0.3806
Epoch 354/500 | Loss: 0.4068
Epoch 355/500 | Loss: 0.3090
Epoch 356/500 | Loss: 0.5491
Epoch 357/500 | Loss: 0.2800
Epoch 358/500 | Loss: 0.3805
Epoch 359/500 | Loss: 0.5357
Epoch 360/500 | Loss: 0.3938
Epoch 361/500 | Loss: 0.7017
Epoch 362/500 | Loss: 0.4421
Epoch 363/500 | Loss: 0.3458
Epoch 364/500 | Loss: 0.5011
Epoch 365/500 | Loss: 0.3452
Epoch 366/500 | Loss: 0.5158
Epoch 367/500 | Loss: 0.4596
Epoch 368/500 | Loss: 0.3963
Epoch 369/500 | Loss: 0.8679
Epoch 370/500 | Loss: 0.3729
Epoch 371/500 | Loss: 0.5928
Epoch 372/500 | Loss: 0.4227
Epoch 373/500 | Loss: 0.3385
Epoch 374/500 | Loss: 0.6785
Epoch 375/500 | Loss: 0.4963
Epoch 376/500 | Loss: 0.4586
Epoch 377/500 | Loss: 0.5066
Epoch 378/500 | Loss: 0.4582
Epoch 379/500 | Loss: 0.2914
Epoch 380/500 | Loss: 0.6431
Epoch 381/500 | Loss: 0.5786
Epoch 382/500 | Loss: 0.4588
Epoch 383/500 | Loss: 0.4748
Epoch 384/500 | Loss: 0.5517
Epoch 385/500 | Loss: 0.3571
Epoch 386/500 | Loss: 0.5328
Epoch 387/500 | Loss: 0.3237
Epoch 388/500 | Loss: 0.6530
Epoch 389/500 | Loss: 0.5302
Epoch 390/500 | Loss: 0.4488
Epoch 391/500 | Loss: 0.3809
Epoch 392/500 | Loss: 0.5672
Epoch 393/500 | Loss: 0.4629
Epoch 394/500 | Loss: 0.6099
Epoch 395/500 | Loss: 0.5862
Epoch 396/500 | Loss: 0.5002
Epoch 397/500 | Loss: 0.4343
Epoch 398/500 | Loss: 0.4525
Epoch 399/500 | Loss: 0.3205
Epoch 400/500 | Loss: 0.3728
Epoch 401/500 | Loss: 0.5327
Epoch 402/500 | Loss: 0.4258
Epoch 403/500 | Loss: 0.4056
Epoch 404/500 | Loss: 0.4016
Epoch 405/500 | Loss: 0.4090
Epoch 406/500 | Loss: 0.4237
Epoch 407/500 | Loss: 0.5472
Epoch 408/500 | Loss: 0.5219
Epoch 409/500 | Loss: 0.4229
Epoch 410/500 | Loss: 0.6281
Epoch 411/500 | Loss: 0.6278
Epoch 412/500 | Loss: 0.5136
Epoch 413/500 | Loss: 0.4503
Epoch 414/500 | Loss: 0.5036
Epoch 415/500 | Loss: 0.4636
Epoch 416/500 | Loss: 0.4617
Epoch 417/500 | Loss: 0.4624
Epoch 418/500 | Loss: 0.3723
Epoch 419/500 | Loss: 0.3053
Epoch 420/500 | Loss: 0.5590
Epoch 421/500 | Loss: 0.3106
Epoch 422/500 | Loss: 0.4483
Epoch 423/500 | Loss: 0.5907
Epoch 424/500 | Loss: 0.3503
Epoch 425/500 | Loss: 0.5299
Epoch 426/500 | Loss: 0.4964
Epoch 427/500 | Loss: 0.4380
Epoch 428/500 | Loss: 0.4836
Epoch 429/500 | Loss: 0.4627
Epoch 430/500 | Loss: 0.3648
Epoch 431/500 | Loss: 0.6110
Epoch 432/500 | Loss: 0.5971
Epoch 433/500 | Loss: 0.6261
Epoch 434/500 | Loss: 0.5027
Epoch 435/500 | Loss: 0.5709
Epoch 436/500 | Loss: 0.5098
Epoch 437/500 | Loss: 0.6106
Epoch 438/500 | Loss: 0.4331
Epoch 439/500 | Loss: 0.3865
Epoch 440/500 | Loss: 0.3410
Epoch 441/500 | Loss: 0.5097
Epoch 442/500 | Loss: 0.4992
Epoch 443/500 | Loss: 0.3974
Epoch 444/500 | Loss: 0.3902
Epoch 445/500 | Loss: 0.3765
Epoch 446/500 | Loss: 0.4409
Epoch 447/500 | Loss: 0.4856
Epoch 448/500 | Loss: 0.5010
Epoch 449/500 | Loss: 0.4425
Epoch 450/500 | Loss: 0.3721
Epoch 451/500 | Loss: 0.5963
Epoch 452/500 | Loss: 0.4473
Epoch 453/500 | Loss: 0.4418
Epoch 454/500 | Loss: 0.3778
Epoch 455/500 | Loss: 0.3348
Epoch 456/500 | Loss: 0.3635
Epoch 457/500 | Loss: 0.5551
Epoch 458/500 | Loss: 0.4711
Epoch 459/500 | Loss: 0.4274
Epoch 460/500 | Loss: 0.4556
Epoch 461/500 | Loss: 0.5802
Epoch 462/500 | Loss: 0.5225
Epoch 463/500 | Loss: 0.6121
Epoch 464/500 | Loss: 0.4126
Epoch 465/500 | Loss: 0.5579
Epoch 466/500 | Loss: 0.5045
Epoch 467/500 | Loss: 0.4046
Epoch 468/500 | Loss: 0.4567
Epoch 469/500 | Loss: 0.4320
Epoch 470/500 | Loss: 0.3594
Epoch 471/500 | Loss: 0.2711
Epoch 472/500 | Loss: 0.5424
Epoch 473/500 | Loss: 0.2819
Epoch 474/500 | Loss: 0.5402
Epoch 475/500 | Loss: 0.4314
Epoch 476/500 | Loss: 0.4278
Epoch 477/500 | Loss: 0.4109
Epoch 478/500 | Loss: 0.5710
Epoch 479/500 | Loss: 0.3343
Epoch 480/500 | Loss: 0.3631
Epoch 481/500 | Loss: 0.5212
Epoch 482/500 | Loss: 0.6754
Epoch 483/500 | Loss: 0.5116
Epoch 484/500 | Loss: 0.3428
Epoch 485/500 | Loss: 0.4414
Epoch 486/500 | Loss: 0.5646
Epoch 487/500 | Loss: 0.5459
Epoch 488/500 | Loss: 0.5536
Epoch 489/500 | Loss: 0.4567
Epoch 490/500 | Loss: 0.3963
Epoch 491/500 | Loss: 0.3855
Epoch 492/500 | Loss: 0.3927
Epoch 493/500 | Loss: 0.3777
Epoch 494/500 | Loss: 0.3421
Epoch 495/500 | Loss: 0.4089
Epoch 496/500 | Loss: 0.2542
Epoch 497/500 | Loss: 0.3854
Epoch 498/500 | Loss: 0.3867
Epoch 499/500 | Loss: 0.6378
Epoch 500/500 | Loss: 0.3370
accuracy: 81.35593220338984
'''
