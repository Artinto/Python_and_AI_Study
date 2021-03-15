import torch #pytorch와 디버깅모듈
import pdb

x_data = [1.0, 2.0, 3.0] #주어진 x,y데이터 셋
y_data = [2.0, 4.0, 6.0]
w = torch.tensor([1.0], requires_grad=True) #기울기값을 tensor로 설정하고 tensor에서 이뤄지는 연산을 추적

# our model forward pass
def forward(x): #일차함수식 계산
    return x * w

# Loss function
def loss(y_pred, y_val): #계산값과 입력값의 차이의 제곱
    return (y_pred - y_val) ** 2

# Before training
print("Prediction (before training)",  4, forward(4).item())

# Training loop
for epoch in range(10): #10번 학습
    for x_val, y_val in zip(x_data, y_data): #각각의 x,y데이터에 대하여
        y_pred = forward(x_val) #forward값 계산
        l = loss(y_pred, y_val) #loss함수의 값 저장 
        l.backward() #tensor(w)의 변화도를 계산(back propagation기법)
        print("\tgrad: ", x_val, y_val, w.grad.item()) #tensor(w)의 변화도는 .grad속성에 누적됨
        w.data = w.data - 0.01 * w.grad.item() #tensor(w)의 값은 .data를 이용하여 접근할 수 있고 w값을 변화시킴 

        # Manually zero the gradients after updating weights
        w.grad.data.zero_() #다음 계산을 위하여 초기화

    print(f"Epoch: {epoch} | Loss: {l.item()}")

# After training
print("Prediction (after training)",  4, forward(4).item())
