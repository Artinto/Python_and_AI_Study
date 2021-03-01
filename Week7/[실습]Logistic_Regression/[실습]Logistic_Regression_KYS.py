import numpy as np
from torch import tensor        # tensor : 계산 속도를 빠르게 하기 위해 GPU를 사용할 수 있게 만들어줌
from torch import nn            # 신경망을 생성할 수 있는 패키지
from torch import sigmoid       # 로지스틱 함수인 sigmoid 사용 
                                # linear model의 결과값을 sigmoid의 입력으로 넣어주면 0-1 사이의 값을 얻음
import torch.nn.functional as F # nn == class / functional == function(객체 생성필요 X)
import torch.optim as optim     # 최적화 알고리즘 정의

dataset_path="/diabetes.csv"
dataset=np.loadtxt(dataset_path, delimiter=',',dtype=np.float32)
x_data=tensor(dataset[:,0:8])
y_data=tensor(dataset[:,8:9])


class Model(nn.Module): # nn.Module을 상속받는  Model class 생성
    def __init__(self): # initial
        super(Model, self).__init__()  # 부모 클래스 생성자 호출
        self.linear = nn.Linear(8, 1)  # One in and one out

    def forward(self, x): # 선형 모델 계산
        y_pred = sigmoid(self.linear(x)) # sigmoid 의 결과값
        return y_pred


model = Model() # Model 객체 생성
criterion = nn.BCELoss(reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=0.01)

cnt = 0
for epoch in range(1001): # == for(i = 0; i < 1001; i++)
    y_pred = model(x_data) # 예측값 저장
    loss = criterion(y_pred, y_data)
    print(f'Epoch {epoch + 1}/1000 | Loss: {loss.item():.4f}')
    if(epoch % 100 == 0):
        for idx in range(len(y_data)):
            if(round(y_pred[idx].item()) == y_data[idx].item()): cnt+=1
        print("Accuracy : ", cnt / len(y_data) * 100)
        cnt = 0
    optimizer.zero_grad() # backword 하기 전 gradient를 초기화 (cuz, gradient가 덮여쓰여지지 않고 누적되기 때문)
    loss.backward()       # loss에 대한 backword 진행
    optimizer.step()      # parameter 초기화

# Epoch 1/1000 | Loss: 0.6945
# Accuracy :  57.57575757575758
# Epoch 2/1000 | Loss: 0.6941
# Epoch 3/1000 | Loss: 0.6938
# Epoch 4/1000 | Loss: 0.6934
# Epoch 5/1000 | Loss: 0.6930
# Epoch 6/1000 | Loss: 0.6927
# Epoch 7/1000 | Loss: 0.6923
# Epoch 8/1000 | Loss: 0.6920
# Epoch 9/1000 | Loss: 0.6916
# Epoch 10/1000 | Loss: 0.6912
# Epoch 11/1000 | Loss: 0.6909
# Epoch 12/1000 | Loss: 0.6906
# Epoch 13/1000 | Loss: 0.6902
# Epoch 14/1000 | Loss: 0.6899
# Epoch 15/1000 | Loss: 0.6895
# Epoch 16/1000 | Loss: 0.6892
# Epoch 17/1000 | Loss: 0.6889
# Epoch 18/1000 | Loss: 0.6885
# Epoch 19/1000 | Loss: 0.6882
# Epoch 20/1000 | Loss: 0.6879
# Epoch 21/1000 | Loss: 0.6875
# Epoch 22/1000 | Loss: 0.6872
# Epoch 23/1000 | Loss: 0.6869
# Epoch 24/1000 | Loss: 0.6866
# Epoch 25/1000 | Loss: 0.6863
# Epoch 26/1000 | Loss: 0.6860
# Epoch 27/1000 | Loss: 0.6857
# Epoch 28/1000 | Loss: 0.6853
# Epoch 29/1000 | Loss: 0.6850
# Epoch 30/1000 | Loss: 0.6847
# Epoch 31/1000 | Loss: 0.6844
# Epoch 32/1000 | Loss: 0.6841
# Epoch 33/1000 | Loss: 0.6838
# Epoch 34/1000 | Loss: 0.6836
# Epoch 35/1000 | Loss: 0.6833
# Epoch 36/1000 | Loss: 0.6830
# Epoch 37/1000 | Loss: 0.6827
# Epoch 38/1000 | Loss: 0.6824
# Epoch 39/1000 | Loss: 0.6821
# Epoch 40/1000 | Loss: 0.6818
# Epoch 41/1000 | Loss: 0.6816
# Epoch 42/1000 | Loss: 0.6813
# Epoch 43/1000 | Loss: 0.6810
# Epoch 44/1000 | Loss: 0.6807
# Epoch 45/1000 | Loss: 0.6805
# Epoch 46/1000 | Loss: 0.6802
# Epoch 47/1000 | Loss: 0.6799
# Epoch 48/1000 | Loss: 0.6797
# Epoch 49/1000 | Loss: 0.6794
# Epoch 50/1000 | Loss: 0.6791
# Epoch 51/1000 | Loss: 0.6789
# Epoch 52/1000 | Loss: 0.6786
# Epoch 53/1000 | Loss: 0.6784
# Epoch 54/1000 | Loss: 0.6781
# Epoch 55/1000 | Loss: 0.6778
# Epoch 56/1000 | Loss: 0.6776
# Epoch 57/1000 | Loss: 0.6773
# Epoch 58/1000 | Loss: 0.6771
# Epoch 59/1000 | Loss: 0.6769
# Epoch 60/1000 | Loss: 0.6766
# Epoch 61/1000 | Loss: 0.6764
# Epoch 62/1000 | Loss: 0.6761
# Epoch 63/1000 | Loss: 0.6759
# Epoch 64/1000 | Loss: 0.6756
# Epoch 65/1000 | Loss: 0.6754
# Epoch 66/1000 | Loss: 0.6752
# Epoch 67/1000 | Loss: 0.6749
# Epoch 68/1000 | Loss: 0.6747
# Epoch 69/1000 | Loss: 0.6745
# Epoch 70/1000 | Loss: 0.6742
# Epoch 71/1000 | Loss: 0.6740
# Epoch 72/1000 | Loss: 0.6738
# Epoch 73/1000 | Loss: 0.6736
# Epoch 74/1000 | Loss: 0.6733
# Epoch 75/1000 | Loss: 0.6731
# Epoch 76/1000 | Loss: 0.6729
# Epoch 77/1000 | Loss: 0.6727
# Epoch 78/1000 | Loss: 0.6725
# Epoch 79/1000 | Loss: 0.6722
# Epoch 80/1000 | Loss: 0.6720
# Epoch 81/1000 | Loss: 0.6718
# Epoch 82/1000 | Loss: 0.6716
# Epoch 83/1000 | Loss: 0.6714
# Epoch 84/1000 | Loss: 0.6712
# Epoch 85/1000 | Loss: 0.6710
# Epoch 86/1000 | Loss: 0.6708
# Epoch 87/1000 | Loss: 0.6705
# Epoch 88/1000 | Loss: 0.6703
# Epoch 89/1000 | Loss: 0.6701
# Epoch 90/1000 | Loss: 0.6699
# Epoch 91/1000 | Loss: 0.6697
# Epoch 92/1000 | Loss: 0.6695
# Epoch 93/1000 | Loss: 0.6693
# Epoch 94/1000 | Loss: 0.6691
# Epoch 95/1000 | Loss: 0.6689
# Epoch 96/1000 | Loss: 0.6687
# Epoch 97/1000 | Loss: 0.6685
# Epoch 98/1000 | Loss: 0.6684
# Epoch 99/1000 | Loss: 0.6682
# Epoch 100/1000 | Loss: 0.6680
# Epoch 101/1000 | Loss: 0.6678
# Accuracy :  64.69038208168642
# Epoch 102/1000 | Loss: 0.6676
# Epoch 103/1000 | Loss: 0.6674
# Epoch 104/1000 | Loss: 0.6672
# Epoch 105/1000 | Loss: 0.6670
# Epoch 106/1000 | Loss: 0.6668
# Epoch 107/1000 | Loss: 0.6667
# Epoch 108/1000 | Loss: 0.6665
# Epoch 109/1000 | Loss: 0.6663
# Epoch 110/1000 | Loss: 0.6661
# Epoch 111/1000 | Loss: 0.6659
# Epoch 112/1000 | Loss: 0.6657
# Epoch 113/1000 | Loss: 0.6656
# Epoch 114/1000 | Loss: 0.6654
# Epoch 115/1000 | Loss: 0.6652
# Epoch 116/1000 | Loss: 0.6650
# Epoch 117/1000 | Loss: 0.6649
# Epoch 118/1000 | Loss: 0.6647
# Epoch 119/1000 | Loss: 0.6645
# Epoch 120/1000 | Loss: 0.6643
# Epoch 121/1000 | Loss: 0.6642
# Epoch 122/1000 | Loss: 0.6640
# Epoch 123/1000 | Loss: 0.6638
# Epoch 124/1000 | Loss: 0.6637
# Epoch 125/1000 | Loss: 0.6635
# Epoch 126/1000 | Loss: 0.6633
# Epoch 127/1000 | Loss: 0.6632
# Epoch 128/1000 | Loss: 0.6630
# Epoch 129/1000 | Loss: 0.6628
# Epoch 130/1000 | Loss: 0.6627
# Epoch 131/1000 | Loss: 0.6625
# Epoch 132/1000 | Loss: 0.6623
# Epoch 133/1000 | Loss: 0.6622
# Epoch 134/1000 | Loss: 0.6620
# Epoch 135/1000 | Loss: 0.6619
# Epoch 136/1000 | Loss: 0.6617
# Epoch 137/1000 | Loss: 0.6615
# Epoch 138/1000 | Loss: 0.6614
# Epoch 139/1000 | Loss: 0.6612
# Epoch 140/1000 | Loss: 0.6611
# Epoch 141/1000 | Loss: 0.6609
# Epoch 142/1000 | Loss: 0.6607
# Epoch 143/1000 | Loss: 0.6606
# Epoch 144/1000 | Loss: 0.6604
# Epoch 145/1000 | Loss: 0.6603
# Epoch 146/1000 | Loss: 0.6601
# Epoch 147/1000 | Loss: 0.6600
# Epoch 148/1000 | Loss: 0.6598
# Epoch 149/1000 | Loss: 0.6597
# Epoch 150/1000 | Loss: 0.6595
# Epoch 151/1000 | Loss: 0.6594
# Epoch 152/1000 | Loss: 0.6592
# Epoch 153/1000 | Loss: 0.6591
# Epoch 154/1000 | Loss: 0.6589
# Epoch 155/1000 | Loss: 0.6588
# Epoch 156/1000 | Loss: 0.6586
# Epoch 157/1000 | Loss: 0.6585
# Epoch 158/1000 | Loss: 0.6584
# Epoch 159/1000 | Loss: 0.6582
# Epoch 160/1000 | Loss: 0.6581
# Epoch 161/1000 | Loss: 0.6579
# Epoch 162/1000 | Loss: 0.6578
# Epoch 163/1000 | Loss: 0.6576
# Epoch 164/1000 | Loss: 0.6575
# Epoch 165/1000 | Loss: 0.6574
# Epoch 166/1000 | Loss: 0.6572
# Epoch 167/1000 | Loss: 0.6571
# Epoch 168/1000 | Loss: 0.6569
# Epoch 169/1000 | Loss: 0.6568
# Epoch 170/1000 | Loss: 0.6567
# Epoch 171/1000 | Loss: 0.6565
# Epoch 172/1000 | Loss: 0.6564
# Epoch 173/1000 | Loss: 0.6563
# Epoch 174/1000 | Loss: 0.6561
# Epoch 175/1000 | Loss: 0.6560
# Epoch 176/1000 | Loss: 0.6558
# Epoch 177/1000 | Loss: 0.6557
# Epoch 178/1000 | Loss: 0.6556
# Epoch 179/1000 | Loss: 0.6554
# Epoch 180/1000 | Loss: 0.6553
# Epoch 181/1000 | Loss: 0.6552
# Epoch 182/1000 | Loss: 0.6550
# Epoch 183/1000 | Loss: 0.6549
# Epoch 184/1000 | Loss: 0.6548
# Epoch 185/1000 | Loss: 0.6547
# Epoch 186/1000 | Loss: 0.6545
# Epoch 187/1000 | Loss: 0.6544
# Epoch 188/1000 | Loss: 0.6543
# Epoch 189/1000 | Loss: 0.6541
# Epoch 190/1000 | Loss: 0.6540
# Epoch 191/1000 | Loss: 0.6539
# Epoch 192/1000 | Loss: 0.6538
# Epoch 193/1000 | Loss: 0.6536
# Epoch 194/1000 | Loss: 0.6535
# Epoch 195/1000 | Loss: 0.6534
# Epoch 196/1000 | Loss: 0.6532
# Epoch 197/1000 | Loss: 0.6531
# Epoch 198/1000 | Loss: 0.6530
# Epoch 199/1000 | Loss: 0.6529
# Epoch 200/1000 | Loss: 0.6527
# Epoch 201/1000 | Loss: 0.6526
# Accuracy :  65.21739130434783
# Epoch 202/1000 | Loss: 0.6525
# Epoch 203/1000 | Loss: 0.6524
# Epoch 204/1000 | Loss: 0.6523
# Epoch 205/1000 | Loss: 0.6521
# Epoch 206/1000 | Loss: 0.6520
# Epoch 207/1000 | Loss: 0.6519
# Epoch 208/1000 | Loss: 0.6518
# Epoch 209/1000 | Loss: 0.6516
# Epoch 210/1000 | Loss: 0.6515
# Epoch 211/1000 | Loss: 0.6514
# Epoch 212/1000 | Loss: 0.6513
# Epoch 213/1000 | Loss: 0.6512
# Epoch 214/1000 | Loss: 0.6511
# Epoch 215/1000 | Loss: 0.6509
# Epoch 216/1000 | Loss: 0.6508
# Epoch 217/1000 | Loss: 0.6507
# Epoch 218/1000 | Loss: 0.6506
# Epoch 219/1000 | Loss: 0.6505
# Epoch 220/1000 | Loss: 0.6503
# Epoch 221/1000 | Loss: 0.6502
# Epoch 222/1000 | Loss: 0.6501
# Epoch 223/1000 | Loss: 0.6500
# Epoch 224/1000 | Loss: 0.6499
# Epoch 225/1000 | Loss: 0.6498
# Epoch 226/1000 | Loss: 0.6497
# Epoch 227/1000 | Loss: 0.6495
# Epoch 228/1000 | Loss: 0.6494
# Epoch 229/1000 | Loss: 0.6493
# Epoch 230/1000 | Loss: 0.6492
# Epoch 231/1000 | Loss: 0.6491
# Epoch 232/1000 | Loss: 0.6490
# Epoch 233/1000 | Loss: 0.6489
# Epoch 234/1000 | Loss: 0.6487
# Epoch 235/1000 | Loss: 0.6486
# Epoch 236/1000 | Loss: 0.6485
# Epoch 237/1000 | Loss: 0.6484
# Epoch 238/1000 | Loss: 0.6483
# Epoch 239/1000 | Loss: 0.6482
# Epoch 240/1000 | Loss: 0.6481
# Epoch 241/1000 | Loss: 0.6480
# Epoch 242/1000 | Loss: 0.6479
# Epoch 243/1000 | Loss: 0.6478
# Epoch 244/1000 | Loss: 0.6476
# Epoch 245/1000 | Loss: 0.6475
# Epoch 246/1000 | Loss: 0.6474
# Epoch 247/1000 | Loss: 0.6473
# Epoch 248/1000 | Loss: 0.6472
# Epoch 249/1000 | Loss: 0.6471
# Epoch 250/1000 | Loss: 0.6470
# Epoch 251/1000 | Loss: 0.6469
# Epoch 252/1000 | Loss: 0.6468
# Epoch 253/1000 | Loss: 0.6467
# Epoch 254/1000 | Loss: 0.6466
# Epoch 255/1000 | Loss: 0.6465
# Epoch 256/1000 | Loss: 0.6463
# Epoch 257/1000 | Loss: 0.6462
# Epoch 258/1000 | Loss: 0.6461
# Epoch 259/1000 | Loss: 0.6460
# Epoch 260/1000 | Loss: 0.6459
# Epoch 261/1000 | Loss: 0.6458
# Epoch 262/1000 | Loss: 0.6457
# Epoch 263/1000 | Loss: 0.6456
# Epoch 264/1000 | Loss: 0.6455
# Epoch 265/1000 | Loss: 0.6454
# Epoch 266/1000 | Loss: 0.6453
# Epoch 267/1000 | Loss: 0.6452
# Epoch 268/1000 | Loss: 0.6451
# Epoch 269/1000 | Loss: 0.6450
# Epoch 270/1000 | Loss: 0.6449
# Epoch 271/1000 | Loss: 0.6448
# Epoch 272/1000 | Loss: 0.6447
# Epoch 273/1000 | Loss: 0.6446
# Epoch 274/1000 | Loss: 0.6445
# Epoch 275/1000 | Loss: 0.6444
# Epoch 276/1000 | Loss: 0.6443
# Epoch 277/1000 | Loss: 0.6442
# Epoch 278/1000 | Loss: 0.6441
# Epoch 279/1000 | Loss: 0.6440
# Epoch 280/1000 | Loss: 0.6439
# Epoch 281/1000 | Loss: 0.6438
# Epoch 282/1000 | Loss: 0.6437
# Epoch 283/1000 | Loss: 0.6436
# Epoch 284/1000 | Loss: 0.6435
# Epoch 285/1000 | Loss: 0.6434
# Epoch 286/1000 | Loss: 0.6433
# Epoch 287/1000 | Loss: 0.6432
# Epoch 288/1000 | Loss: 0.6431
# Epoch 289/1000 | Loss: 0.6430
# Epoch 290/1000 | Loss: 0.6429
# Epoch 291/1000 | Loss: 0.6428
# Epoch 292/1000 | Loss: 0.6427
# Epoch 293/1000 | Loss: 0.6426
# Epoch 294/1000 | Loss: 0.6425
# Epoch 295/1000 | Loss: 0.6424
# Epoch 296/1000 | Loss: 0.6423
# Epoch 297/1000 | Loss: 0.6422
# Epoch 298/1000 | Loss: 0.6421
# Epoch 299/1000 | Loss: 0.6420
# Epoch 300/1000 | Loss: 0.6419
# Epoch 301/1000 | Loss: 0.6418
# Accuracy :  65.34914361001317
# Epoch 302/1000 | Loss: 0.6417
# Epoch 303/1000 | Loss: 0.6416
# Epoch 304/1000 | Loss: 0.6415
# Epoch 305/1000 | Loss: 0.6414
# Epoch 306/1000 | Loss: 0.6413
# Epoch 307/1000 | Loss: 0.6412
# Epoch 308/1000 | Loss: 0.6411
# Epoch 309/1000 | Loss: 0.6410
# Epoch 310/1000 | Loss: 0.6409
# Epoch 311/1000 | Loss: 0.6408
# Epoch 312/1000 | Loss: 0.6407
# Epoch 313/1000 | Loss: 0.6407
# Epoch 314/1000 | Loss: 0.6406
# Epoch 315/1000 | Loss: 0.6405
# Epoch 316/1000 | Loss: 0.6404
# Epoch 317/1000 | Loss: 0.6403
# Epoch 318/1000 | Loss: 0.6402
# Epoch 319/1000 | Loss: 0.6401
# Epoch 320/1000 | Loss: 0.6400
# Epoch 321/1000 | Loss: 0.6399
# Epoch 322/1000 | Loss: 0.6398
# Epoch 323/1000 | Loss: 0.6397
# Epoch 324/1000 | Loss: 0.6396
# Epoch 325/1000 | Loss: 0.6395
# Epoch 326/1000 | Loss: 0.6394
# Epoch 327/1000 | Loss: 0.6393
# Epoch 328/1000 | Loss: 0.6393
# Epoch 329/1000 | Loss: 0.6392
# Epoch 330/1000 | Loss: 0.6391
# Epoch 331/1000 | Loss: 0.6390
# Epoch 332/1000 | Loss: 0.6389
# Epoch 333/1000 | Loss: 0.6388
# Epoch 334/1000 | Loss: 0.6387
# Epoch 335/1000 | Loss: 0.6386
# Epoch 336/1000 | Loss: 0.6385
# Epoch 337/1000 | Loss: 0.6384
# Epoch 338/1000 | Loss: 0.6383
# Epoch 339/1000 | Loss: 0.6382
# Epoch 340/1000 | Loss: 0.6382
# Epoch 341/1000 | Loss: 0.6381
# Epoch 342/1000 | Loss: 0.6380
# Epoch 343/1000 | Loss: 0.6379
# Epoch 344/1000 | Loss: 0.6378
# Epoch 345/1000 | Loss: 0.6377
# Epoch 346/1000 | Loss: 0.6376
# Epoch 347/1000 | Loss: 0.6375
# Epoch 348/1000 | Loss: 0.6374
# Epoch 349/1000 | Loss: 0.6373
# Epoch 350/1000 | Loss: 0.6373
# Epoch 351/1000 | Loss: 0.6372
# Epoch 352/1000 | Loss: 0.6371
# Epoch 353/1000 | Loss: 0.6370
# Epoch 354/1000 | Loss: 0.6369
# Epoch 355/1000 | Loss: 0.6368
# Epoch 356/1000 | Loss: 0.6367
# Epoch 357/1000 | Loss: 0.6366
# Epoch 358/1000 | Loss: 0.6365
# Epoch 359/1000 | Loss: 0.6365
# Epoch 360/1000 | Loss: 0.6364
# Epoch 361/1000 | Loss: 0.6363
# Epoch 362/1000 | Loss: 0.6362
# Epoch 363/1000 | Loss: 0.6361
# Epoch 364/1000 | Loss: 0.6360
# Epoch 365/1000 | Loss: 0.6359
# Epoch 366/1000 | Loss: 0.6358
# Epoch 367/1000 | Loss: 0.6358
# Epoch 368/1000 | Loss: 0.6357
# Epoch 369/1000 | Loss: 0.6356
# Epoch 370/1000 | Loss: 0.6355
# Epoch 371/1000 | Loss: 0.6354
# Epoch 372/1000 | Loss: 0.6353
# Epoch 373/1000 | Loss: 0.6352
# Epoch 374/1000 | Loss: 0.6351
# Epoch 375/1000 | Loss: 0.6351
# Epoch 376/1000 | Loss: 0.6350
# Epoch 377/1000 | Loss: 0.6349
# Epoch 378/1000 | Loss: 0.6348
# Epoch 379/1000 | Loss: 0.6347
# Epoch 380/1000 | Loss: 0.6346
# Epoch 381/1000 | Loss: 0.6345
# Epoch 382/1000 | Loss: 0.6345
# Epoch 383/1000 | Loss: 0.6344
# Epoch 384/1000 | Loss: 0.6343
# Epoch 385/1000 | Loss: 0.6342
# Epoch 386/1000 | Loss: 0.6341
# Epoch 387/1000 | Loss: 0.6340
# Epoch 388/1000 | Loss: 0.6339
# Epoch 389/1000 | Loss: 0.6339
# Epoch 390/1000 | Loss: 0.6338
# Epoch 391/1000 | Loss: 0.6337
# Epoch 392/1000 | Loss: 0.6336
# Epoch 393/1000 | Loss: 0.6335
# Epoch 394/1000 | Loss: 0.6334
# Epoch 395/1000 | Loss: 0.6333
# Epoch 396/1000 | Loss: 0.6333
# Epoch 397/1000 | Loss: 0.6332
# Epoch 398/1000 | Loss: 0.6331
# Epoch 399/1000 | Loss: 0.6330
# Epoch 400/1000 | Loss: 0.6329
# Epoch 401/1000 | Loss: 0.6328
# Accuracy :  65.34914361001317
# Epoch 402/1000 | Loss: 0.6328
# Epoch 403/1000 | Loss: 0.6327
# Epoch 404/1000 | Loss: 0.6326
# Epoch 405/1000 | Loss: 0.6325
# Epoch 406/1000 | Loss: 0.6324
# Epoch 407/1000 | Loss: 0.6323
# Epoch 408/1000 | Loss: 0.6323
# Epoch 409/1000 | Loss: 0.6322
# Epoch 410/1000 | Loss: 0.6321
# Epoch 411/1000 | Loss: 0.6320
# Epoch 412/1000 | Loss: 0.6319
# Epoch 413/1000 | Loss: 0.6318
# Epoch 414/1000 | Loss: 0.6318
# Epoch 415/1000 | Loss: 0.6317
# Epoch 416/1000 | Loss: 0.6316
# Epoch 417/1000 | Loss: 0.6315
# Epoch 418/1000 | Loss: 0.6314
# Epoch 419/1000 | Loss: 0.6313
# Epoch 420/1000 | Loss: 0.6313
# Epoch 421/1000 | Loss: 0.6312
# Epoch 422/1000 | Loss: 0.6311
# Epoch 423/1000 | Loss: 0.6310
# Epoch 424/1000 | Loss: 0.6309
# Epoch 425/1000 | Loss: 0.6309
# Epoch 426/1000 | Loss: 0.6308
# Epoch 427/1000 | Loss: 0.6307
# Epoch 428/1000 | Loss: 0.6306
# Epoch 429/1000 | Loss: 0.6305
# Epoch 430/1000 | Loss: 0.6305
# Epoch 431/1000 | Loss: 0.6304
# Epoch 432/1000 | Loss: 0.6303
# Epoch 433/1000 | Loss: 0.6302
# Epoch 434/1000 | Loss: 0.6301
# Epoch 435/1000 | Loss: 0.6300
# Epoch 436/1000 | Loss: 0.6300
# Epoch 437/1000 | Loss: 0.6299
# Epoch 438/1000 | Loss: 0.6298
# Epoch 439/1000 | Loss: 0.6297
# Epoch 440/1000 | Loss: 0.6296
# Epoch 441/1000 | Loss: 0.6296
# Epoch 442/1000 | Loss: 0.6295
# Epoch 443/1000 | Loss: 0.6294
# Epoch 444/1000 | Loss: 0.6293
# Epoch 445/1000 | Loss: 0.6292
# Epoch 446/1000 | Loss: 0.6292
# Epoch 447/1000 | Loss: 0.6291
# Epoch 448/1000 | Loss: 0.6290
# Epoch 449/1000 | Loss: 0.6289
# Epoch 450/1000 | Loss: 0.6288
# Epoch 451/1000 | Loss: 0.6288
# Epoch 452/1000 | Loss: 0.6287
# Epoch 453/1000 | Loss: 0.6286
# Epoch 454/1000 | Loss: 0.6285
# Epoch 455/1000 | Loss: 0.6284
# Epoch 456/1000 | Loss: 0.6284
# Epoch 457/1000 | Loss: 0.6283
# Epoch 458/1000 | Loss: 0.6282
# Epoch 459/1000 | Loss: 0.6281
# Epoch 460/1000 | Loss: 0.6280
# Epoch 461/1000 | Loss: 0.6280
# Epoch 462/1000 | Loss: 0.6279
# Epoch 463/1000 | Loss: 0.6278
# Epoch 464/1000 | Loss: 0.6277
# Epoch 465/1000 | Loss: 0.6277
# Epoch 466/1000 | Loss: 0.6276
# Epoch 467/1000 | Loss: 0.6275
# Epoch 468/1000 | Loss: 0.6274
# Epoch 469/1000 | Loss: 0.6273
# Epoch 470/1000 | Loss: 0.6273
# Epoch 471/1000 | Loss: 0.6272
# Epoch 472/1000 | Loss: 0.6271
# Epoch 473/1000 | Loss: 0.6270
# Epoch 474/1000 | Loss: 0.6270
# Epoch 475/1000 | Loss: 0.6269
# Epoch 476/1000 | Loss: 0.6268
# Epoch 477/1000 | Loss: 0.6267
# Epoch 478/1000 | Loss: 0.6266
# Epoch 479/1000 | Loss: 0.6266
# Epoch 480/1000 | Loss: 0.6265
# Epoch 481/1000 | Loss: 0.6264
# Epoch 482/1000 | Loss: 0.6263
# Epoch 483/1000 | Loss: 0.6263
# Epoch 484/1000 | Loss: 0.6262
# Epoch 485/1000 | Loss: 0.6261
# Epoch 486/1000 | Loss: 0.6260
# Epoch 487/1000 | Loss: 0.6259
# Epoch 488/1000 | Loss: 0.6259
# Epoch 489/1000 | Loss: 0.6258
# Epoch 490/1000 | Loss: 0.6257
# Epoch 491/1000 | Loss: 0.6256
# Epoch 492/1000 | Loss: 0.6256
# Epoch 493/1000 | Loss: 0.6255
# Epoch 494/1000 | Loss: 0.6254
# Epoch 495/1000 | Loss: 0.6253
# Epoch 496/1000 | Loss: 0.6253
# Epoch 497/1000 | Loss: 0.6252
# Epoch 498/1000 | Loss: 0.6251
# Epoch 499/1000 | Loss: 0.6250
# Epoch 500/1000 | Loss: 0.6249
# Epoch 501/1000 | Loss: 0.6249
# Accuracy :  65.34914361001317
# Epoch 502/1000 | Loss: 0.6248
# Epoch 503/1000 | Loss: 0.6247
# Epoch 504/1000 | Loss: 0.6246
# Epoch 505/1000 | Loss: 0.6246
# Epoch 506/1000 | Loss: 0.6245
# Epoch 507/1000 | Loss: 0.6244
# Epoch 508/1000 | Loss: 0.6243
# Epoch 509/1000 | Loss: 0.6243
# Epoch 510/1000 | Loss: 0.6242
# Epoch 511/1000 | Loss: 0.6241
# Epoch 512/1000 | Loss: 0.6240
# Epoch 513/1000 | Loss: 0.6240
# Epoch 514/1000 | Loss: 0.6239
# Epoch 515/1000 | Loss: 0.6238
# Epoch 516/1000 | Loss: 0.6237
# Epoch 517/1000 | Loss: 0.6237
# Epoch 518/1000 | Loss: 0.6236
# Epoch 519/1000 | Loss: 0.6235
# Epoch 520/1000 | Loss: 0.6234
# Epoch 521/1000 | Loss: 0.6234
# Epoch 522/1000 | Loss: 0.6233
# Epoch 523/1000 | Loss: 0.6232
# Epoch 524/1000 | Loss: 0.6231
# Epoch 525/1000 | Loss: 0.6231
# Epoch 526/1000 | Loss: 0.6230
# Epoch 527/1000 | Loss: 0.6229
# Epoch 528/1000 | Loss: 0.6228
# Epoch 529/1000 | Loss: 0.6228
# Epoch 530/1000 | Loss: 0.6227
# Epoch 531/1000 | Loss: 0.6226
# Epoch 532/1000 | Loss: 0.6225
# Epoch 533/1000 | Loss: 0.6225
# Epoch 534/1000 | Loss: 0.6224
# Epoch 535/1000 | Loss: 0.6223
# Epoch 536/1000 | Loss: 0.6222
# Epoch 537/1000 | Loss: 0.6222
# Epoch 538/1000 | Loss: 0.6221
# Epoch 539/1000 | Loss: 0.6220
# Epoch 540/1000 | Loss: 0.6219
# Epoch 541/1000 | Loss: 0.6219
# Epoch 542/1000 | Loss: 0.6218
# Epoch 543/1000 | Loss: 0.6217
# Epoch 544/1000 | Loss: 0.6217
# Epoch 545/1000 | Loss: 0.6216
# Epoch 546/1000 | Loss: 0.6215
# Epoch 547/1000 | Loss: 0.6214
# Epoch 548/1000 | Loss: 0.6214
# Epoch 549/1000 | Loss: 0.6213
# Epoch 550/1000 | Loss: 0.6212
# Epoch 551/1000 | Loss: 0.6211
# Epoch 552/1000 | Loss: 0.6211
# Epoch 553/1000 | Loss: 0.6210
# Epoch 554/1000 | Loss: 0.6209
# Epoch 555/1000 | Loss: 0.6208
# Epoch 556/1000 | Loss: 0.6208
# Epoch 557/1000 | Loss: 0.6207
# Epoch 558/1000 | Loss: 0.6206
# Epoch 559/1000 | Loss: 0.6206
# Epoch 560/1000 | Loss: 0.6205
# Epoch 561/1000 | Loss: 0.6204
# Epoch 562/1000 | Loss: 0.6203
# Epoch 563/1000 | Loss: 0.6203
# Epoch 564/1000 | Loss: 0.6202
# Epoch 565/1000 | Loss: 0.6201
# Epoch 566/1000 | Loss: 0.6200
# Epoch 567/1000 | Loss: 0.6200
# Epoch 568/1000 | Loss: 0.6199
# Epoch 569/1000 | Loss: 0.6198
# Epoch 570/1000 | Loss: 0.6198
# Epoch 571/1000 | Loss: 0.6197
# Epoch 572/1000 | Loss: 0.6196
# Epoch 573/1000 | Loss: 0.6195
# Epoch 574/1000 | Loss: 0.6195
# Epoch 575/1000 | Loss: 0.6194
# Epoch 576/1000 | Loss: 0.6193
# Epoch 577/1000 | Loss: 0.6193
# Epoch 578/1000 | Loss: 0.6192
# Epoch 579/1000 | Loss: 0.6191
# Epoch 580/1000 | Loss: 0.6190
# Epoch 581/1000 | Loss: 0.6190
# Epoch 582/1000 | Loss: 0.6189
# Epoch 583/1000 | Loss: 0.6188
# Epoch 584/1000 | Loss: 0.6188
# Epoch 585/1000 | Loss: 0.6187
# Epoch 586/1000 | Loss: 0.6186
# Epoch 587/1000 | Loss: 0.6185
# Epoch 588/1000 | Loss: 0.6185
# Epoch 589/1000 | Loss: 0.6184
# Epoch 590/1000 | Loss: 0.6183
# Epoch 591/1000 | Loss: 0.6183
# Epoch 592/1000 | Loss: 0.6182
# Epoch 593/1000 | Loss: 0.6181
# Epoch 594/1000 | Loss: 0.6180
# Epoch 595/1000 | Loss: 0.6180
# Epoch 596/1000 | Loss: 0.6179
# Epoch 597/1000 | Loss: 0.6178
# Epoch 598/1000 | Loss: 0.6178
# Epoch 599/1000 | Loss: 0.6177
# Epoch 600/1000 | Loss: 0.6176
# Epoch 601/1000 | Loss: 0.6175
# Accuracy :  65.34914361001317
# Epoch 602/1000 | Loss: 0.6175
# Epoch 603/1000 | Loss: 0.6174
# Epoch 604/1000 | Loss: 0.6173
# Epoch 605/1000 | Loss: 0.6173
# Epoch 606/1000 | Loss: 0.6172
# Epoch 607/1000 | Loss: 0.6171
# Epoch 608/1000 | Loss: 0.6171
# Epoch 609/1000 | Loss: 0.6170
# Epoch 610/1000 | Loss: 0.6169
# Epoch 611/1000 | Loss: 0.6168
# Epoch 612/1000 | Loss: 0.6168
# Epoch 613/1000 | Loss: 0.6167
# Epoch 614/1000 | Loss: 0.6166
# Epoch 615/1000 | Loss: 0.6166
# Epoch 616/1000 | Loss: 0.6165
# Epoch 617/1000 | Loss: 0.6164
# Epoch 618/1000 | Loss: 0.6164
# Epoch 619/1000 | Loss: 0.6163
# Epoch 620/1000 | Loss: 0.6162
# Epoch 621/1000 | Loss: 0.6161
# Epoch 622/1000 | Loss: 0.6161
# Epoch 623/1000 | Loss: 0.6160
# Epoch 624/1000 | Loss: 0.6159
# Epoch 625/1000 | Loss: 0.6159
# Epoch 626/1000 | Loss: 0.6158
# Epoch 627/1000 | Loss: 0.6157
# Epoch 628/1000 | Loss: 0.6157
# Epoch 629/1000 | Loss: 0.6156
# Epoch 630/1000 | Loss: 0.6155
# Epoch 631/1000 | Loss: 0.6155
# Epoch 632/1000 | Loss: 0.6154
# Epoch 633/1000 | Loss: 0.6153
# Epoch 634/1000 | Loss: 0.6152
# Epoch 635/1000 | Loss: 0.6152
# Epoch 636/1000 | Loss: 0.6151
# Epoch 637/1000 | Loss: 0.6150
# Epoch 638/1000 | Loss: 0.6150
# Epoch 639/1000 | Loss: 0.6149
# Epoch 640/1000 | Loss: 0.6148
# Epoch 641/1000 | Loss: 0.6148
# Epoch 642/1000 | Loss: 0.6147
# Epoch 643/1000 | Loss: 0.6146
# Epoch 644/1000 | Loss: 0.6146
# Epoch 645/1000 | Loss: 0.6145
# Epoch 646/1000 | Loss: 0.6144
# Epoch 647/1000 | Loss: 0.6144
# Epoch 648/1000 | Loss: 0.6143
# Epoch 649/1000 | Loss: 0.6142
# Epoch 650/1000 | Loss: 0.6142
# Epoch 651/1000 | Loss: 0.6141
# Epoch 652/1000 | Loss: 0.6140
# Epoch 653/1000 | Loss: 0.6139
# Epoch 654/1000 | Loss: 0.6139
# Epoch 655/1000 | Loss: 0.6138
# Epoch 656/1000 | Loss: 0.6137
# Epoch 657/1000 | Loss: 0.6137
# Epoch 658/1000 | Loss: 0.6136
# Epoch 659/1000 | Loss: 0.6135
# Epoch 660/1000 | Loss: 0.6135
# Epoch 661/1000 | Loss: 0.6134
# Epoch 662/1000 | Loss: 0.6133
# Epoch 663/1000 | Loss: 0.6133
# Epoch 664/1000 | Loss: 0.6132
# Epoch 665/1000 | Loss: 0.6131
# Epoch 666/1000 | Loss: 0.6131
# Epoch 667/1000 | Loss: 0.6130
# Epoch 668/1000 | Loss: 0.6129
# Epoch 669/1000 | Loss: 0.6129
# Epoch 670/1000 | Loss: 0.6128
# Epoch 671/1000 | Loss: 0.6127
# Epoch 672/1000 | Loss: 0.6127
# Epoch 673/1000 | Loss: 0.6126
# Epoch 674/1000 | Loss: 0.6125
# Epoch 675/1000 | Loss: 0.6125
# Epoch 676/1000 | Loss: 0.6124
# Epoch 677/1000 | Loss: 0.6123
# Epoch 678/1000 | Loss: 0.6123
# Epoch 679/1000 | Loss: 0.6122
# Epoch 680/1000 | Loss: 0.6121
# Epoch 681/1000 | Loss: 0.6121
# Epoch 682/1000 | Loss: 0.6120
# Epoch 683/1000 | Loss: 0.6119
# Epoch 684/1000 | Loss: 0.6119
# Epoch 685/1000 | Loss: 0.6118
# Epoch 686/1000 | Loss: 0.6117
# Epoch 687/1000 | Loss: 0.6117
# Epoch 688/1000 | Loss: 0.6116
# Epoch 689/1000 | Loss: 0.6115
# Epoch 690/1000 | Loss: 0.6115
# Epoch 691/1000 | Loss: 0.6114
# Epoch 692/1000 | Loss: 0.6113
# Epoch 693/1000 | Loss: 0.6113
# Epoch 694/1000 | Loss: 0.6112
# Epoch 695/1000 | Loss: 0.6111
# Epoch 696/1000 | Loss: 0.6111
# Epoch 697/1000 | Loss: 0.6110
# Epoch 698/1000 | Loss: 0.6109
# Epoch 699/1000 | Loss: 0.6109
# Epoch 700/1000 | Loss: 0.6108
# Epoch 701/1000 | Loss: 0.6107
# Accuracy :  65.21739130434783
# Epoch 702/1000 | Loss: 0.6107
# Epoch 703/1000 | Loss: 0.6106
# Epoch 704/1000 | Loss: 0.6105
# Epoch 705/1000 | Loss: 0.6105
# Epoch 706/1000 | Loss: 0.6104
# Epoch 707/1000 | Loss: 0.6103
# Epoch 708/1000 | Loss: 0.6103
# Epoch 709/1000 | Loss: 0.6102
# Epoch 710/1000 | Loss: 0.6101
# Epoch 711/1000 | Loss: 0.6101
# Epoch 712/1000 | Loss: 0.6100
# Epoch 713/1000 | Loss: 0.6099
# Epoch 714/1000 | Loss: 0.6099
# Epoch 715/1000 | Loss: 0.6098
# Epoch 716/1000 | Loss: 0.6098
# Epoch 717/1000 | Loss: 0.6097
# Epoch 718/1000 | Loss: 0.6096
# Epoch 719/1000 | Loss: 0.6096
# Epoch 720/1000 | Loss: 0.6095
# Epoch 721/1000 | Loss: 0.6094
# Epoch 722/1000 | Loss: 0.6094
# Epoch 723/1000 | Loss: 0.6093
# Epoch 724/1000 | Loss: 0.6092
# Epoch 725/1000 | Loss: 0.6092
# Epoch 726/1000 | Loss: 0.6091
# Epoch 727/1000 | Loss: 0.6090
# Epoch 728/1000 | Loss: 0.6090
# Epoch 729/1000 | Loss: 0.6089
# Epoch 730/1000 | Loss: 0.6088
# Epoch 731/1000 | Loss: 0.6088
# Epoch 732/1000 | Loss: 0.6087
# Epoch 733/1000 | Loss: 0.6087
# Epoch 734/1000 | Loss: 0.6086
# Epoch 735/1000 | Loss: 0.6085
# Epoch 736/1000 | Loss: 0.6085
# Epoch 737/1000 | Loss: 0.6084
# Epoch 738/1000 | Loss: 0.6083
# Epoch 739/1000 | Loss: 0.6083
# Epoch 740/1000 | Loss: 0.6082
# Epoch 741/1000 | Loss: 0.6081
# Epoch 742/1000 | Loss: 0.6081
# Epoch 743/1000 | Loss: 0.6080
# Epoch 744/1000 | Loss: 0.6079
# Epoch 745/1000 | Loss: 0.6079
# Epoch 746/1000 | Loss: 0.6078
# Epoch 747/1000 | Loss: 0.6078
# Epoch 748/1000 | Loss: 0.6077
# Epoch 749/1000 | Loss: 0.6076
# Epoch 750/1000 | Loss: 0.6076
# Epoch 751/1000 | Loss: 0.6075
# Epoch 752/1000 | Loss: 0.6074
# Epoch 753/1000 | Loss: 0.6074
# Epoch 754/1000 | Loss: 0.6073
# Epoch 755/1000 | Loss: 0.6072
# Epoch 756/1000 | Loss: 0.6072
# Epoch 757/1000 | Loss: 0.6071
# Epoch 758/1000 | Loss: 0.6071
# Epoch 759/1000 | Loss: 0.6070
# Epoch 760/1000 | Loss: 0.6069
# Epoch 761/1000 | Loss: 0.6069
# Epoch 762/1000 | Loss: 0.6068
# Epoch 763/1000 | Loss: 0.6067
# Epoch 764/1000 | Loss: 0.6067
# Epoch 765/1000 | Loss: 0.6066
# Epoch 766/1000 | Loss: 0.6065
# Epoch 767/1000 | Loss: 0.6065
# Epoch 768/1000 | Loss: 0.6064
# Epoch 769/1000 | Loss: 0.6064
# Epoch 770/1000 | Loss: 0.6063
# Epoch 771/1000 | Loss: 0.6062
# Epoch 772/1000 | Loss: 0.6062
# Epoch 773/1000 | Loss: 0.6061
# Epoch 774/1000 | Loss: 0.6060
# Epoch 775/1000 | Loss: 0.6060
# Epoch 776/1000 | Loss: 0.6059
# Epoch 777/1000 | Loss: 0.6059
# Epoch 778/1000 | Loss: 0.6058
# Epoch 779/1000 | Loss: 0.6057
# Epoch 780/1000 | Loss: 0.6057
# Epoch 781/1000 | Loss: 0.6056
# Epoch 782/1000 | Loss: 0.6055
# Epoch 783/1000 | Loss: 0.6055
# Epoch 784/1000 | Loss: 0.6054
# Epoch 785/1000 | Loss: 0.6054
# Epoch 786/1000 | Loss: 0.6053
# Epoch 787/1000 | Loss: 0.6052
# Epoch 788/1000 | Loss: 0.6052
# Epoch 789/1000 | Loss: 0.6051
# Epoch 790/1000 | Loss: 0.6050
# Epoch 791/1000 | Loss: 0.6050
# Epoch 792/1000 | Loss: 0.6049
# Epoch 793/1000 | Loss: 0.6049
# Epoch 794/1000 | Loss: 0.6048
# Epoch 795/1000 | Loss: 0.6047
# Epoch 796/1000 | Loss: 0.6047
# Epoch 797/1000 | Loss: 0.6046
# Epoch 798/1000 | Loss: 0.6045
# Epoch 799/1000 | Loss: 0.6045
# Epoch 800/1000 | Loss: 0.6044
# Epoch 801/1000 | Loss: 0.6044
# Accuracy :  65.21739130434783
# Epoch 802/1000 | Loss: 0.6043
# Epoch 803/1000 | Loss: 0.6042
# Epoch 804/1000 | Loss: 0.6042
# Epoch 805/1000 | Loss: 0.6041
# Epoch 806/1000 | Loss: 0.6041
# Epoch 807/1000 | Loss: 0.6040
# Epoch 808/1000 | Loss: 0.6039
# Epoch 809/1000 | Loss: 0.6039
# Epoch 810/1000 | Loss: 0.6038
# Epoch 811/1000 | Loss: 0.6037
# Epoch 812/1000 | Loss: 0.6037
# Epoch 813/1000 | Loss: 0.6036
# Epoch 814/1000 | Loss: 0.6036
# Epoch 815/1000 | Loss: 0.6035
# Epoch 816/1000 | Loss: 0.6034
# Epoch 817/1000 | Loss: 0.6034
# Epoch 818/1000 | Loss: 0.6033
# Epoch 819/1000 | Loss: 0.6033
# Epoch 820/1000 | Loss: 0.6032
# Epoch 821/1000 | Loss: 0.6031
# Epoch 822/1000 | Loss: 0.6031
# Epoch 823/1000 | Loss: 0.6030
# Epoch 824/1000 | Loss: 0.6030
# Epoch 825/1000 | Loss: 0.6029
# Epoch 826/1000 | Loss: 0.6028
# Epoch 827/1000 | Loss: 0.6028
# Epoch 828/1000 | Loss: 0.6027
# Epoch 829/1000 | Loss: 0.6026
# Epoch 830/1000 | Loss: 0.6026
# Epoch 831/1000 | Loss: 0.6025
# Epoch 832/1000 | Loss: 0.6025
# Epoch 833/1000 | Loss: 0.6024
# Epoch 834/1000 | Loss: 0.6023
# Epoch 835/1000 | Loss: 0.6023
# Epoch 836/1000 | Loss: 0.6022
# Epoch 837/1000 | Loss: 0.6022
# Epoch 838/1000 | Loss: 0.6021
# Epoch 839/1000 | Loss: 0.6020
# Epoch 840/1000 | Loss: 0.6020
# Epoch 841/1000 | Loss: 0.6019
# Epoch 842/1000 | Loss: 0.6019
# Epoch 843/1000 | Loss: 0.6018
# Epoch 844/1000 | Loss: 0.6017
# Epoch 845/1000 | Loss: 0.6017
# Epoch 846/1000 | Loss: 0.6016
# Epoch 847/1000 | Loss: 0.6016
# Epoch 848/1000 | Loss: 0.6015
# Epoch 849/1000 | Loss: 0.6014
# Epoch 850/1000 | Loss: 0.6014
# Epoch 851/1000 | Loss: 0.6013
# Epoch 852/1000 | Loss: 0.6013
# Epoch 853/1000 | Loss: 0.6012
# Epoch 854/1000 | Loss: 0.6011
# Epoch 855/1000 | Loss: 0.6011
# Epoch 856/1000 | Loss: 0.6010
# Epoch 857/1000 | Loss: 0.6010
# Epoch 858/1000 | Loss: 0.6009
# Epoch 859/1000 | Loss: 0.6008
# Epoch 860/1000 | Loss: 0.6008
# Epoch 861/1000 | Loss: 0.6007
# Epoch 862/1000 | Loss: 0.6007
# Epoch 863/1000 | Loss: 0.6006
# Epoch 864/1000 | Loss: 0.6006
# Epoch 865/1000 | Loss: 0.6005
# Epoch 866/1000 | Loss: 0.6004
# Epoch 867/1000 | Loss: 0.6004
# Epoch 868/1000 | Loss: 0.6003
# Epoch 869/1000 | Loss: 0.6003
# Epoch 870/1000 | Loss: 0.6002
# Epoch 871/1000 | Loss: 0.6001
# Epoch 872/1000 | Loss: 0.6001
# Epoch 873/1000 | Loss: 0.6000
# Epoch 874/1000 | Loss: 0.6000
# Epoch 875/1000 | Loss: 0.5999
# Epoch 876/1000 | Loss: 0.5998
# Epoch 877/1000 | Loss: 0.5998
# Epoch 878/1000 | Loss: 0.5997
# Epoch 879/1000 | Loss: 0.5997
# Epoch 880/1000 | Loss: 0.5996
# Epoch 881/1000 | Loss: 0.5995
# Epoch 882/1000 | Loss: 0.5995
# Epoch 883/1000 | Loss: 0.5994
# Epoch 884/1000 | Loss: 0.5994
# Epoch 885/1000 | Loss: 0.5993
# Epoch 886/1000 | Loss: 0.5993
# Epoch 887/1000 | Loss: 0.5992
# Epoch 888/1000 | Loss: 0.5991
# Epoch 889/1000 | Loss: 0.5991
# Epoch 890/1000 | Loss: 0.5990
# Epoch 891/1000 | Loss: 0.5990
# Epoch 892/1000 | Loss: 0.5989
# Epoch 893/1000 | Loss: 0.5988
# Epoch 894/1000 | Loss: 0.5988
# Epoch 895/1000 | Loss: 0.5987
# Epoch 896/1000 | Loss: 0.5987
# Epoch 897/1000 | Loss: 0.5986
# Epoch 898/1000 | Loss: 0.5986
# Epoch 899/1000 | Loss: 0.5985
# Epoch 900/1000 | Loss: 0.5984
# Epoch 901/1000 | Loss: 0.5984
# Accuracy :  65.87615283267458
# Epoch 902/1000 | Loss: 0.5983
# Epoch 903/1000 | Loss: 0.5983
# Epoch 904/1000 | Loss: 0.5982
# Epoch 905/1000 | Loss: 0.5982
# Epoch 906/1000 | Loss: 0.5981
# Epoch 907/1000 | Loss: 0.5980
# Epoch 908/1000 | Loss: 0.5980
# Epoch 909/1000 | Loss: 0.5979
# Epoch 910/1000 | Loss: 0.5979
# Epoch 911/1000 | Loss: 0.5978
# Epoch 912/1000 | Loss: 0.5977
# Epoch 913/1000 | Loss: 0.5977
# Epoch 914/1000 | Loss: 0.5976
# Epoch 915/1000 | Loss: 0.5976
# Epoch 916/1000 | Loss: 0.5975
# Epoch 917/1000 | Loss: 0.5975
# Epoch 918/1000 | Loss: 0.5974
# Epoch 919/1000 | Loss: 0.5973
# Epoch 920/1000 | Loss: 0.5973
# Epoch 921/1000 | Loss: 0.5972
# Epoch 922/1000 | Loss: 0.5972
# Epoch 923/1000 | Loss: 0.5971
# Epoch 924/1000 | Loss: 0.5971
# Epoch 925/1000 | Loss: 0.5970
# Epoch 926/1000 | Loss: 0.5969
# Epoch 927/1000 | Loss: 0.5969
# Epoch 928/1000 | Loss: 0.5968
# Epoch 929/1000 | Loss: 0.5968
# Epoch 930/1000 | Loss: 0.5967
# Epoch 931/1000 | Loss: 0.5967
# Epoch 932/1000 | Loss: 0.5966
# Epoch 933/1000 | Loss: 0.5965
# Epoch 934/1000 | Loss: 0.5965
# Epoch 935/1000 | Loss: 0.5964
# Epoch 936/1000 | Loss: 0.5964
# Epoch 937/1000 | Loss: 0.5963
# Epoch 938/1000 | Loss: 0.5963
# Epoch 939/1000 | Loss: 0.5962
# Epoch 940/1000 | Loss: 0.5962
# Epoch 941/1000 | Loss: 0.5961
# Epoch 942/1000 | Loss: 0.5960
# Epoch 943/1000 | Loss: 0.5960
# Epoch 944/1000 | Loss: 0.5959
# Epoch 945/1000 | Loss: 0.5959
# Epoch 946/1000 | Loss: 0.5958
# Epoch 947/1000 | Loss: 0.5958
# Epoch 948/1000 | Loss: 0.5957
# Epoch 949/1000 | Loss: 0.5956
# Epoch 950/1000 | Loss: 0.5956
# Epoch 951/1000 | Loss: 0.5955
# Epoch 952/1000 | Loss: 0.5955
# Epoch 953/1000 | Loss: 0.5954
# Epoch 954/1000 | Loss: 0.5954
# Epoch 955/1000 | Loss: 0.5953
# Epoch 956/1000 | Loss: 0.5953
# Epoch 957/1000 | Loss: 0.5952
# Epoch 958/1000 | Loss: 0.5951
# Epoch 959/1000 | Loss: 0.5951
# Epoch 960/1000 | Loss: 0.5950
# Epoch 961/1000 | Loss: 0.5950
# Epoch 962/1000 | Loss: 0.5949
# Epoch 963/1000 | Loss: 0.5949
# Epoch 964/1000 | Loss: 0.5948
# Epoch 965/1000 | Loss: 0.5947
# Epoch 966/1000 | Loss: 0.5947
# Epoch 967/1000 | Loss: 0.5946
# Epoch 968/1000 | Loss: 0.5946
# Epoch 969/1000 | Loss: 0.5945
# Epoch 970/1000 | Loss: 0.5945
# Epoch 971/1000 | Loss: 0.5944
# Epoch 972/1000 | Loss: 0.5944
# Epoch 973/1000 | Loss: 0.5943
# Epoch 974/1000 | Loss: 0.5942
# Epoch 975/1000 | Loss: 0.5942
# Epoch 976/1000 | Loss: 0.5941
# Epoch 977/1000 | Loss: 0.5941
# Epoch 978/1000 | Loss: 0.5940
# Epoch 979/1000 | Loss: 0.5940
# Epoch 980/1000 | Loss: 0.5939
# Epoch 981/1000 | Loss: 0.5939
# Epoch 982/1000 | Loss: 0.5938
# Epoch 983/1000 | Loss: 0.5938
# Epoch 984/1000 | Loss: 0.5937
# Epoch 985/1000 | Loss: 0.5936
# Epoch 986/1000 | Loss: 0.5936
# Epoch 987/1000 | Loss: 0.5935
# Epoch 988/1000 | Loss: 0.5935
# Epoch 989/1000 | Loss: 0.5934
# Epoch 990/1000 | Loss: 0.5934
# Epoch 991/1000 | Loss: 0.5933
# Epoch 992/1000 | Loss: 0.5933
# Epoch 993/1000 | Loss: 0.5932
# Epoch 994/1000 | Loss: 0.5931
# Epoch 995/1000 | Loss: 0.5931
# Epoch 996/1000 | Loss: 0.5930
# Epoch 997/1000 | Loss: 0.5930
# Epoch 998/1000 | Loss: 0.5929
# Epoch 999/1000 | Loss: 0.5929
# Epoch 1000/1000 | Loss: 0.5928
# Epoch 1001/1000 | Loss: 0.5928
# Accuracy :  65.61264822134387