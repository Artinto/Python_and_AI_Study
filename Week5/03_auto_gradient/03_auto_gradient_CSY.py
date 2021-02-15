# pytorch는 python을 위한 오픈소스 머신러닝 라이브러리로 torch 모듈을 가져오면 자동 미분 기능을 사용할 수 있음
# pdb(Python Debugger) 모듈은 python 프로그램을 위한 대화형 소스코드 디버거로 이 모듈을 가져옴 
import torch
import pdb

# input 데이터인 x data와 결과값인 y data를 리스트에 저장
# torch.tensor로 tensor을 생성하여 w에 1.0을 저장하고 requires_grad 속성을 True로 설정하여 tensor에서 이뤄진 모든 연산들을 추적하도록 함
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
w = torch.tensor([1.0], requires_grad=True)

# our model forward pass
# 인자 x를 받는 순방향 진행 함수로 선형 함수인 x * w를 반환
def forward(x):
    return x * w

# Loss function
# 손실함수로 인자 y_pred와  y_val를 받으며 y_pred값과 y_val값과의 차이를 제곱한 값을 반환함(양수여야 하기 때문) 
def loss(y_pred, y_val):
    return (y_pred - y_val) ** 2

# Before training
# 훈련 전의 예측값은 input값인 x에 4를 대입했을 때 결과값으로 4가 나온다는 것을 출력(w값이 1.0이므로)
# .item()은 tensor 속의 숫자를 스칼라 값으로 반환하는 함수로 forward(4)의 스칼라 값을 반환함
print("Prediction (before training)",  4, forward(4).item())

# Training loop
# 반복 훈련 과정으로 for문을 이용하여 10번(0~9) 반복함
# x_val와 y_val에 x_data와 y_data를 차례로 대입함( zip 함수는 동일한 개수로 이루어진 자료형을 묶어 주는 역할을 하는 함수)
for epoch in range(10):
    for x_val, y_val in zip(x_data, y_data):
        # x_val값을 인자로 넘겨 forward함수를 통해 반환된 값을 y_pred에 대입(선형 모델 식을 통해 나온 예측 y값)
        # x_val와 y_val를 loss함수의 인자로 넘겨 반환된 값(손실함수 값)을 l에 저장
        # l.backward()를 통해 back propagation(역전파)를 실행
        # x_val, y_val와 loss함수 기울기 값인 w.grad의 스칼라 값 출력
        # w.data에 loss함수를 미분해서 얻은 기울기 값인 w.grad에 alpha값인 0.01을 곱한 값을 빼준 후 그 값을 w.data에 대입하여 weight값을 업데이트 시킴
        y_pred = forward(x_val) # 1) Forward pass
        l = loss(y_pred, y_val) # 2) Compute loss
        l.backward() # 3) Back propagation to update weights
        print("\tgrad: ", x_val, y_val, w.grad.item())
        w.data = w.data - 0.01 * w.grad.item()

        # Manually zero the gradients after updating weights
        # weight를 갱신 후에 수동으로 변화도를 0으로 만듦(.backward()를 호출할 때마다 변화도가 버퍼에 덮어씌워지지 않고 누적되기 때문)
        w.grad.data.zero_()

    # 훈련 반복 횟수인 epoch 값과 loss 값 출력
    print(f"Epoch: {epoch} | Loss: {l.item()}")

# After training
# 훈련 후에 input 값을 4로 했을 때 결과값을 출력
print("Prediction (after training)",  4, forward(4).item())
