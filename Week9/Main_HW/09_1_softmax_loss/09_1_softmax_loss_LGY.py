# 09_01_softmax_loss.py
from torch import nn, tensor, max
# nn > CrossEntropyLoss()
# 인덱스 뽑아주는 max함수

import numpy as np

# One hot
Y = np.array([1, 0, 0])
Y_pred1 = np.array([0.7, 0.2, 0.1]) # 확률로 표현된
Y_pred2 = np.array([0.1, 0.3, 0.6])
print(f'Loss1: {np.sum(-Y * np.log(Y_pred1)):.4f}') # cross-entropy수식
print(f'Loss2: {np.sum(-Y * np.log(Y_pred2)):.4f}')

# Softmax + CrossEntropy (따라서 input은 softmax하기 전, value 값이어야 한다.)
loss = nn.CrossEntropyLoss()
# Cross entropy 식
# ex) Y = [1, 0, 0] <=> Y_pred= [0.7, 0.2, 0.1]
#     >> -[1*log(0.7) + 0*log(0.2) + 0*log(0.1)] = -log(0.7) = 0.35

'''
CrossEntropyLoss() >> return F.cross_entropy >> return nll_loss >>
'''

# target is of size nBatch
# each element in target has to have 0 <= value < nClasses (0-2)
# Input is class, not one-hot
Y = tensor([0], requires_grad=False) # 클래스 (0, 1, 2 중에 답이 0)

# input is of size nBatch x nClasses = 1 x 4
# Y_pred are logits (not softmax)
Y_pred1 = tensor([[2.0, 1.0, 0.1]]) # scores(logit)
Y_pred2 = tensor([[0.5, 2.0, 0.3]])


l1 = loss(Y_pred1, Y) # nn.CrossEntropyLoss() = softmax + crossentropy
l2 = loss(Y_pred2, Y) # inputs > (minibatch, class label)
# nn.CrossEntropyLoss()는 Y(정답)를 one-hot vector로 넘겨줄 필요 없이 그냥 class label로 주면 됨.
# reduction = 'mean'이라서 알아서 loss 평균 내줌.

print(f'PyTorch Loss1: {l1.item():.4f} \nPyTorch Loss2: {l2.item():.4f}')
print(f'Y_pred1: {max(Y_pred1.data, 1)[1].item()}')
print(f'Y_pred2: {max(Y_pred2.data, 1)[1].item()}')
# torch.max :주어진 텐서 배열의 최대 값이 들어있는 index를 리턴하는 함수 (axis = 1)
#torch.max함수 : (values, indices)의 튜플을 return해 줌.
'''
# max(Y_pred1.data, 1)
: torch.return_types.max(values=tensor([2.]),indices=tensor([0]))
'''

# target is of size nBatch
Y = tensor([2, 0, 1], requires_grad=False)
# 위에서는 한문제, 여기서는 3문제 (역시 one-hot이 아닌 class label로 주어짐)

# input is of size nBatch x nClasses = 2 x 4
Y_pred1 = tensor([[0.1, 0.2, 0.9],
                  [1.1, 0.1, 0.2],
                  [0.2, 2.1, 0.1]]) # 3문제, logits(scores)

Y_pred2 = tensor([[0.8, 0.2, 0.3],
                  [0.2, 0.3, 0.5],
                  [0.2, 0.2, 0.5]])

l1 = loss(Y_pred1, Y) # softmax + crossentropy
l2 = loss(Y_pred2, Y)
print(f'Batch Loss1:  {l1.item():.4f} \nBatch Loss2: {l2.data:.4f}')


