import torch  # pytorch 호출
import pdb  # pdb(파이썬 디버거) 호출

x_data = [1.0, 2.0, 3.0]  # x데이터 입력
y_data = [2.0, 4.0, 6.0]  # y데이터 입력
w = torch.tensor([1.0], requires_grad=True) # 값이 1.0인 1x1의 2차원 텐서 w를 선언하고, 이 텐서에 기울기를 저장하겠다

# our model forward pass
def forward(x): # 포워드 함수 선언(x*w)
    return x * w

# Loss function
def loss(y_pred, y_val):  # 오차 함수 선언
    return (y_pred - y_val) ** 2

# Before training
print("Prediction (before training)",  4, forward(4).item())  # 기계 학습 전 대입. --> 무의미한 값이 나와야함

# Training loop
for epoch in range(10): # 10번 경사 하강 
    for x_val, y_val in zip(x_data, y_data):  # x_val, y_val에 각각 x_data, y_data의 쌍들을 집어넣음
        y_pred = forward(x_val) # 1) Forward pass
        l = loss(y_pred, y_val) # 2) Compute loss
        l.backward() # 3) Back propagation to update weights, loss함수를 이용하여 기울기 계산
        print("\tgrad: ", x_val, y_val, w.grad.item())  # x_val, y_val, 기울기를 출력
        w.data = w.data - 0.01 * w.grad.item()  # w값에 따른 w-cost함수의 기울기가 음수면 오른쪽, 양수면 왼쪽으로 가도록 이동 --> 경사를 따라 내려가게됌 (학습률 0.01)

        # Manually zero the gradients after updating weights
        w.grad.data.zero_() #미분값(기울기)의 누적을 방지하기 위해 미분데이터를 0으로 초기화 시킴.

    print(f"Epoch: {epoch} | Loss: {l.item()}") # 경사하강법의 반복횟수와 오차값을 출력

# After training
print("Prediction (after training)",  4, forward(4).item()) # 기계 학습 후 대입. ->> 기존 데이터와 비례하는 값이 나와야함
