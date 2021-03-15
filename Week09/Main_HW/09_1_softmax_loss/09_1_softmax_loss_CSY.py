# torch 모듈의 필요한 패키지 불러옴
# numpy 패키지 불러옴
from torch import nn, tensor, max
import numpy as np

# Cross entropy example
# One hot
# 0: 1 0 0
# 1: 0 1 0
# 2: 0 0 1
# 결과값인 Y를 numpy배열로 선언(one-hot lables)
# 예측값인 Y_pred1,2를 선언
# cross entropy 손실함수를 만들어 각각의 예측값의 손실값을 계산
Y = np.array([1, 0, 0])
Y_pred1 = np.array([0.7, 0.2, 0.1])
Y_pred2 = np.array([0.1, 0.3, 0.6])
print(f'Loss1: {np.sum(-Y * np.log(Y_pred1)):.4f}')
print(f'Loss2: {np.sum(-Y * np.log(Y_pred2)):.4f}')

# Softmax + CrossEntropy (logSoftmax + NLLLoss)
# nn.CrossEntropyLoss 함수는 Softmax와 Cross Entropy가 합쳐진 것이며 때문에 Y_pred 값에 softmax할 필요 없음
loss = nn.CrossEntropyLoss()

# target is of size nBatch
# each element in target has to have 0 <= value < nClasses (0-2)
# Input is class, not one-hot
# 0번째 클래스를 의미/역전파 중에 이 Tensor에 대한 변화도를 계산할 필요가 없음을 나타냄
Y = tensor([0], requires_grad=False)

# input is of size nBatch x nClasses = 1 x 4
# Y_pred are logits (not softmax)
# Y_pred1,2에 softmax를 하지 않은 Logit값 저장
Y_pred1 = tensor([[2.0, 1.0, 0.1]])
Y_pred2 = tensor([[0.5, 2.0, 0.3]])

# Y_Pred와 Y에 대한 손실값 계산
l1 = loss(Y_pred1, Y)
l2 = loss(Y_pred2, Y)

# 각 예측에 대한 loss값 출력
# 각 예측값에서의 최대값의 인덱스 출력
print(f'PyTorch Loss1: {l1.item():.4f} \nPyTorch Loss2: {l2.item():.4f}')
print(f'Y_pred1: {max(Y_pred1.data, 1)[1].item()}')
print(f'Y_pred2: {max(Y_pred2.data, 1)[1].item()}')

# target is of size nBatch
# each element in target has to have 0 <= value < nClasses (0-2)
# Input is class, not one-hot
# 여러 개의 Y클래스 선언/역전파 중에 이 Tensor에 대한 변화도를 계산할 필요가 없음을 나타냄
Y = tensor([2, 0, 1], requires_grad=False)

# input is of size nBatch x nClasses = 2 x 4
# Y_pred are logits (not softmax)
# Y_pred1,2에 softmax를 하지 않은 Logit값 저장
Y_pred1 = tensor([[0.1, 0.2, 0.9],
                  [1.1, 0.1, 0.2],
                  [0.2, 2.1, 0.1]])

Y_pred2 = tensor([[0.8, 0.2, 0.3],
                  [0.2, 0.3, 0.5],
                  [0.2, 0.2, 0.5]])

# Y_Pred와 Y에 대한 손실값 계산
# 각 예측에 대한 loss값 출력
l1 = loss(Y_pred1, Y)
l2 = loss(Y_pred2, Y)
print(f'Batch Loss1:  {l1.item():.4f} \nBatch Loss2: {l2.data:.4f}')
