from torch import nn, tensor, max # torch에서 nn, tensor, max를 불러옴
import numpy as np  #numpy를 np라 선언

# Cross entropy example
# One hot
# 0: 1 0 0
# 1: 0 1 0
# 2: 0 0 1
Y = np.array([1, 0, 0]) # Y를 [1, 0, 0]의 1차원 넘파이 배열로 선언
Y_pred1 = np.array([0.7, 0.2, 0.1]) # Y_pred1을 [0.7, 0.2, 0.1]의 1차원 넘파이 배열로 선언
Y_pred2 = np.array([0.1, 0.3, 0.6]) # Y_pred1를 [0.1, 0.3, 0.6]의 1차원 넘파이 배열로 선언
print(f'Loss1: {np.sum(-Y * np.log(Y_pred1)):.4f}') # Loss1의 값으로 Y_pred1과 Y를 가지고 CrossEntropy 함수를 실행하여 오차값을 계산후 출력
print(f'Loss2: {np.sum(-Y * np.log(Y_pred2)):.4f}') # Loss2의 값으로 Y_pred2과 Y를 가지고 CrossEntropy 함수를 실행하여 오차값을 계산후 출력

# Softmax + CrossEntropy (logSoftmax + NLLLoss)
loss = nn.CrossEntropyLoss()  # 오차 loss에 nn.CrossEntropyLoss()할당. CrossEntropyLoss에는 Softmax 함수와 CrossEntropy함수가 들어 있음

# target is of size nBatch
# each element in target has to have 0 <= value < nClasses (0-2)
# Input is class, not one-hot
Y = tensor([0], requires_grad=False)  # Y는 0,1,2 중에서 0의 원-핫인코딩벡터, 즉 [1, 0, 0]

# input is of size nBatch x nClasses = 1 x 4
# Y_pred are logits (not softmax)
Y_pred1 = tensor([[2.0, 1.0, 0.1]]) # Y_pred1을 [[2.0, 1.0, 0.1]]의 1x3의 2차원 매트릭스로 지정
Y_pred2 = tensor([[0.5, 2.0, 0.3]]) # Y_pred2를 [[0.5, 2.0, 0.3]]의 1x3의 2차원 매트릭스로 지정

l1 = loss(Y_pred1, Y) # l1을 Y_pred1과 Y의 오차로 지정
l2 = loss(Y_pred2, Y) # l2를 Y_pred2와 Y의 오차로 지정

print(f'PyTorch Loss1: {l1.item():.4f} \nPyTorch Loss2: {l2.item():.4f}') # l1과 l2값 출력
print(f'Y_pred1: {max(Y_pred1.data, 1)[1].item()}') 
print(f'Y_pred2: {max(Y_pred2.data, 1)[1].item()}')

# target is of size nBatch
# each element in target has to have 0 <= value < nClasses (0-2)
# Input is class, not one-hot
Y = tensor([2, 0, 1], requires_grad=False)  # Y는 0,1,2 중에서 각각 2, 0, 1의 원-핫인코딩벡터들, 즉 [0, 0, 1],[1, 0, 0],[0, 1, 0]

# input is of size nBatch x nClasses = 2 x 4
# Y_pred are logits (not softmax)
Y_pred1 = tensor([[0.1, 0.2, 0.9],
                  [1.1, 0.1, 0.2],
                  [0.2, 2.1, 0.1]]) # Y_pred1을 3x3의 2차원 매트릭스로 지정

Y_pred2 = tensor([[0.8, 0.2, 0.3], 
                  [0.2, 0.3, 0.5],
                  [0.2, 0.2, 0.5]]) # Y_pred2를 3x3의 2차원 매트릭스로 지정

l1 = loss(Y_pred1, Y) # l1을 Y_pred1과 Y의 오차로 지정
l2 = loss(Y_pred2, Y) # l2를 Y_pred2와 Y의 오차로 지정
print(f'Batch Loss1:  {l1.item():.4f} \nBatch Loss2: {l2.data:.4f}')  # loss1과 loss2의 값 출력
