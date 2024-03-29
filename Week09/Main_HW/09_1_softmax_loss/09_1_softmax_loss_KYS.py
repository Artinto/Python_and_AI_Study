from torch import nn, tensor, max # max : tensor안의 최댓값 index return
import numpy as np

# Cross entropy example
# One hot
# 0: 1 0 0
# 1: 0 1 0
# 2: 0 0 1
Y = np.array([1, 0, 0]) # One hot 벡터 > 0 index(class)의 정답을 확인하기 위해
Y_pred1 = np.array([0.7, 0.2, 0.1]) # softmax의 결과값 지정(right answer)
Y_pred2 = np.array([0.1, 0.3, 0.6]) # softmax의 결과값 지정(wrong answer)
print(f'Loss1: {np.sum(-Y * np.log(Y_pred1)):.4f}')
print(f'Loss2: {np.sum(-Y * np.log(Y_pred2)):.4f}')

# Softmax + CrossEntropy (logSoftmax + NLLLoss)
loss = nn.CrossEntropyLoss() # CrossEntropyLoss 사용

# target is of size nBatch
# each element in target has to have 0 <= value < nClasses (0-2)
# Input is class, not one-hot
Y = tensor([0], requires_grad=False) # 첫번째 class([1,0,0]) 예측

# input is of size nBatch x nClasses = 1 x 4
# Y_pred are logits (not softmax)
Y_pred1 = tensor([[2.0, 1.0, 0.1]]) # 임의의 예측값 tensor에 저장
Y_pred2 = tensor([[0.5, 2.0, 0.3]]) # softmax 사용 이전 데이터

l1 = loss(Y_pred1, Y) # loss 구하기
l2 = loss(Y_pred2, Y)

print(f'PyTorch Loss1: {l1.item():.4f} \nPyTorch Loss2: {l2.item():.4f}')
print(f'Y_pred1: {max(Y_pred1.data, 1)[1].item()}')
print(f'Y_pred2: {max(Y_pred2.data, 1)[1].item()}')

# target is of size nBatch
# each element in target has to have 0 <= value < nClasses (0-2)
# Input is class, not one-hot
Y = tensor([2, 0, 1], requires_grad=False) # One hot 벡터의 2,0,1번재 클래스 예측

# input is of size nBatch x nClasses = 2 x 4
# Y_pred are logits (not softmax)
Y_pred1 = tensor([[0.1, 0.2, 0.9], # One hot 벡터와 행렬곱을 하기 위해 3x3으로 tensor list 생성
                  [1.1, 0.1, 0.2],
                  [0.2, 2.1, 0.1]])

Y_pred2 = tensor([[0.8, 0.2, 0.3],
                  [0.2, 0.3, 0.5],
                  [0.2, 0.2, 0.5]])

l1 = loss(Y_pred1, Y) # loss 구하기
l2 = loss(Y_pred2, Y)
print(f'Batch Loss1:  {l1.item():.4f} \nBatch Loss2: {l2.data:.4f}')