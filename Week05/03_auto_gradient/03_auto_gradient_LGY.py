import torch # torch > autograd > backward
import pdb    # 코드에서 사용하지 않고 있으나 python 디버깅할 때 사용하는 모듈로
# pdb.set_trace()을 중단하고 싶은 곳에 넣으면 실행이 중지된다

# PDB관련참고링크 : http://pythonstudy.xyz/python/article/505-Python-%EB%94%94%EB%B2%84%EA%B9%85-PDB

#데이터 선언
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# initial weight
w = torch.tensor([1.0], requires_grad=True) 
# requires_grad=True : backpropagation 중 이 tensor의 변화도를 계산하라
# default 값 : True

# Pytorch 0.4 이후로는 Tensor가 Variable을 완전히 대체하여 Variable없이 Tensor로 구축한다고 함.
# >>Variable선언없이 tensor만으로 가능해짐.
# print(torch.__version__) # 1.7.0+cu101


# model : Linear
def forward(x):
    return x * w

# Loss function
def loss(y_pred, y_val): 
    return (y_pred - y_val) ** 2 #(예측 - 실제)^2

# 학습이 하나도 되지 않았을 때의 예측값
print("Prediction (before training)",  4, forward(4).item()) # 모델에 x = 4가 들어갔을 때 return값(w*x) //(w : 초기값 0.1)

# training 과정
for epoch in range(10): # 10번 반복학습시키겠다 (forward, backward를 10번 반복)
    for x_val, y_val in zip(x_data, y_data):
        y_pred = forward(x_val) # 1) Forward pass
        l = loss(y_pred, y_val) # 2) loss func을 통해 loss값 구하기
        l.backward() # 3) Back propagation to update weights
        # w가 거쳐온 모든 노드의 gradient를 계산하여 w.grad에 저장함.
        print("\tgrad: ", x_val, y_val, w.grad.item()) # w.grad는 tensor형. tensor형을 print해주기 위해서 item()로 스칼라 값을 가져온다
        w.data = w.data - 0.01 * w.grad.item() # update
        # w는 저장소 / w.data는 value / w.grad = (loss변화량/w변화량)

        # backward()함수는 이전 gradient에 누적하여 계산되기 때문에 다음 계산을 위해 초기화시켜줌.
        w.grad.data.zero_()

    print(f"Epoch: {epoch} | Loss: {l.item()}")

# 학습이 끝난 후 모델의 예측값
print("Prediction (after training)",  4, forward(4).item())
