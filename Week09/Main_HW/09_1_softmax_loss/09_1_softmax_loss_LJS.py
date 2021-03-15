import numpy as np
import torch

Y = np.array([1,0,0]) # One - Hot Label 

y_pred1 = np.array([0.7, 0.2, 0.1])
y_pred2 = np.array([0.1, 0.3, 0.6])

# Multi - Class Classification : SoftMax + Cross Entropy 
# Binary - Class Classification : Sigmoid + Cross Entropy 

print(f'loss1 = { np.sum(-Y * np.log( y_pred1 )):.4f}')
print(f'loss2 = { np.sum(-Y * np.log( y_pred2 )):.4f}')


loss = torch.nn.CrossEntropyLoss()

Y = torch.tensor([0], requires_grad = False) # Class  ex) 1번 클래스, 2번 클래스 ...

#  == [1,0,0]

Y_pred1 = torch.tensor([ [2.0, 1.0, 0.1] ])
Y_pred2 = torch.tensor([ [0.5, 2.0, 0.3] ])

l1 = loss(Y_pred1, Y) # Loss 함수의 내부적으로 Softmax 를 통과한다.
l2 = loss(Y_pred2, Y)

# Softmax Loss 출력
print(f'Pytorch Loss1 : {l1.item():.4f} \n Pytorch Loss2 : {l2.item():.4f}' )

print(f'Y_pred1: {torch.max(Y_pred1.data, 1)[1].item()}')
print(f'Y_pred2: {torch.max(Y_pred2.data, 1)[1].item()}')

# torch.max(Y_pred1, 1) # return 값은 1과 element 중 큰 값의 Index, [1]으로 그 Index만 취한다.

Y = torch.tensor([2,0,1], requires_grad=False ) # 여러 개 Target Class


# 분류를 진행할 3개의 Input tensor

Y_pred1 = torch.tensor([[0.1, 0.2, 0.9], 
                  [1.1, 0.1, 0.2],
                  [0.2, 2.1, 0.1]])

Y_pred2 = torch.tensor([[0.8, 0.2, 0.3],
                  [0.2, 0.3, 0.5],
                  [0.2, 0.2, 0.5]])


# 내부적으로 Loss 함수는 Softmax 함수를 통과시킨다.

l1 = loss(Y_pred1, Y)
l2 = loss(Y_pred2, Y)

print(f'batch Loss1 : {l1.item():.4f} \n Batch Loss2 : {l2.item():.4f}')


      







