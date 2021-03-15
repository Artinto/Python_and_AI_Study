import torch# tensor, grad 함수를 쓰기위한 모듈
import pdb # 디버깅도구로 줄마다 보면서 실행할수있는 도

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
w = torch.tensor([1.0], requires_grad=True)# 행렬과 벡터를 포함하는걸 텐서라고 하고 w를 1로 지정하고 머신러닝으로  기울기를 자동으로 구하게함

# our model forward pass
def forward(x): #x * w
    return x * w

# Loss function
def loss(y_pred, y_val):  #손실함수 구하는 식으로 (yhead-y)**2
    return (y_pred - y_val) ** 2

# Before training
print("Prediction (before training)",  4, forward(4).item()) #forward에 4를대입 ==4

# Training loop
for epoch in range(10): #10번 실행 실행수가 많을수록 정확
    for x_val, y_val in zip(x_data, y_data):
        y_pred = forward(x_val) # 1) Forward pass
        l = loss(y_pred, y_val) # 2) Compute loss
        l.backward() # 3) Back propagation to update weights   loss=(y'-y)**2     >>>>>>>>> loss미분=2w(2wx-y)   >>>>>>> 이때의 w x y 값으로 loss미분값을 구함 이값이 0에 가까워지는 w값이 정확한  값이된다
        print("\tgrad: ", x_val, y_val, w.grad.item())#x,y의 값과 w기울기의 미분값을 프린트
        w.data = w.data - 0.01 * w.grad.item()#소수점이 낮을수록 정확

        # Manually zero the gradients after updating weights
        w.grad.data.zero_() #w미분 데이터를 0으로 만듬(초기화)

    print(f"Epoch: {epoch} | Loss: {l.item()}")# for문 반복횟수 와 손실함수의 값을 말해줌

# After training
print("Prediction (after training)",  4, forward(4).item())
