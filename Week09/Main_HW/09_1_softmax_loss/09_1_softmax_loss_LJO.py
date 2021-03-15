from torch import nn, tensor, max
import numpy as np

# Cross entropy example
# One hot
# 0: 1 0 0
# 1: 0 1 0
# 2: 0 0 1
Y = np.array([1, 0, 0]) #one hot 벡터
Y_pred1 = np.array([0.7, 0.2, 0.1]) #y예측값 여기서는 softmax를 거침
Y_pred2 = np.array([0.1, 0.3, 0.6])
print(f'Loss1: {np.sum(-Y * np.log(Y_pred1)):.4f}') #loss값 출력 -Ylog(Y_pred)
print(f'Loss2: {np.sum(-Y * np.log(Y_pred2)):.4f}')

# Softmax + CrossEntropy (logSoftmax + NLLLoss)
loss = nn.CrossEntropyLoss() # logSoftmax와 NLLLoss의 조합_pred2 = np.array([0.1, 0.3, 0.6])
print(f'Loss1: {np.sum(-Y * np.log(Y_pred1)):.4f}') #loss값 출력 -Ylog(Y_pred)
print(f'Loss2: {np.sum(-Y * np.log(Y_pred2)):.4f}')

# Softmax + CrossEntropy (logSoftmax + NLLLoss)
loss = nn.CrossEntropyLoss() # logSoftmax와 NLLLoss의 조합 logSoftmax는 Softmax의 로그값, NLLLoss는 교차 엔트로피 손실 연산oss = nn.CrossEntropyLoss() # logSoftmax와 NLLLoss의 조합 logSoftmax는 Softmax의 로그값, NLLLoss는 교차 엔트로피 손실 연산

# target is of size nBatch
# each element in target has to have 0 <= value < nClasses (0-2)
# Input is class, not one-hot
Y = tensor([0], requires_grad=False) #Y값 0~2, 클래스(값 0이아니라 0번쨰를 의미)

# input is of size nBatch x nClasses = 1 x 4
# Y_pred are logits (not softmax)
Y_pred1 = tensor([[2.0, 1.0, 0.1]]) #y예측값 softmax를 거치지않음
Y_pred2 = tensor([[0.5, 2.0, 0.3]])

l1 = loss(Y_pred1, Y) #loss값 계산
l2 = loss(Y_pred2, Y)

print(f'PyTorch Loss1: {l1.item():.4f} \nPyTorch Loss2: {l2.item():.4f}') #loss값과 y예측값의 최댓값 인텍스 출력
print(f'Y_pred1: {max(Y_pred1.data, 1)[1].item()}')
print(f'Y_pred2: {max(Y_pred2.data, 1)[1].item()}')

# target is of size nBatch
# each element in target has to have 0 <= value < nClasses (0-2)
# Input is class, not one-hot
Y = tensor([2, 0, 1], requires_grad=False) #여러개의 y값

# input is of size nBatch x nClasses = 2 x 4
# Y_pred are logits (not softmax)
Y_pred1 = tensor([[0.1, 0.2, 0.9], #y_pred또한 multiple
                  [1.1, 0.1, 0.2],
                  [0.2, 2.1, 0.1]])

Y_pred2 = tensor([[0.8, 0.2, 0.3],
                  [0.2, 0.3, 0.5],
                  [0.2, 0.2, 0.5]])

l1 = loss(Y_pred1, Y) #y예측값 softmax를 거치지않음
l2 = loss(Y_pred2, Y)
print(f'Batch Loss1:  {l1.item():.4f} \nBatch Loss2: {l2.data:.4f}')  #loss값 출력
