from torch import nn, tensor, max
import numpy as np


# Cross entropy example
# One hot
# 0: 1 0 0
# 1: 0 1 0
# 2: 0 0 1
Y = np.array([1, 0, 0]) #각 행렬에 인덱싱을 부여해주기 위하여 one hot encoding을 실행
Y_pred1 = np.array([0.7, 0.2, 0.1]) # softmax를 계산한 값을 임의로 대입
Y_pred2 = np.array([0.1, 0.3, 0.6]) # 잘못된 값 비교용 대입
print(f'Loss1: {np.sum(-Y * np.log(Y_pred1)):.4f}') # 그 값에 대하여 크로스 엔트로피를 실행
print(f'Loss2: {np.sum(-Y * np.log(Y_pred2)):.4f}')

# Softmax + CrossEntropy (logSoftmax + NLLLoss)
loss = nn.CrossEntropyLoss() #크로스엔트로피 내장 함수

# target is of size nBatch
# each element in target has to have 0 <= value < nClasses (0-2)
# Input is class, not one-hot
Y = tensor([0], requires_grad=False) #0번째 예측

# input is of size nBatch x nClasses = 1 x 4
# Y_pred are logits (not softmax)
Y_pred1 = tensor([[2.0, 1.0, 0.1]]) #올바른 예측값
Y_pred2 = tensor([[0.5, 2.0, 0.3]]) #비교용 틀린 예측값

l1 = loss(Y_pred1, Y)
l2 = loss(Y_pred2, Y)

print(f'PyTorch Loss1: {l1.item():.4f} \n PyTorch Loss2: {l2.item():.4f}')
print(f'Y_pred1: {max(Y_pred1.data, 1)[1].item()}') # max(data, 1)[1]은 인덱스(자리의 위치)만 추출 item은 텐서안에서 숫자만 빼오는 것
print(f'Y_pred2: {max(Y_pred2.data, 1)[1].item()}')

# target is of size nBatch
# each element in target has to have 0 <= value < nClasses (0-2)
# Input is class, not one-hot
Y = tensor([2, 0, 1], requires_grad=False) #역전파에 대한 변화도 계산 필요 X이기에 False

# input is of size nBatch x nClasses = 2 x 4
# Y_pred are logits (not softmax)
Y_pred1 = tensor([[0.1, 0.2, 0.9],
                  [1.1, 0.1, 0.2],
                  [0.2, 2.1, 0.1]]) # 3x3

Y_pred2 = tensor([[0.8, 0.2, 0.3],
                  [0.2, 0.3, 0.5],
                  [0.2, 0.2, 0.5]])

l1 = loss(Y_pred1, Y)
l2 = loss(Y_pred2, Y)
print(f'Batch Loss1:  {l1.item():.4f} \nBatch Loss2: {l2.data:.4f}') # Loss에 대한 값 출력
#soft_max값이 아닌 그냥 값을 넣어주는 이유 loss에서 soft_max를 실행하여 계산해줌