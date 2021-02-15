import torch#tensor함수를 가지고 있는 모듈로 최대한의 유연성과 속도 제공 
import pdb#대화형 디버거를 제공하는 모듈
#
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
w = torch.tensor([1.0], requires_grad=True)#tensor를 통해 배열을 만들며 requires_grad를 하여 자동미분이 되도록 만든다

# our model forward pass
def forward(x):
    return x * w

# Loss function
def loss(y_pred, y_val):
    return (y_pred - y_val) ** 2

# Before training
print("Prediction (before training)",  4, forward(4).item())

# Training loop
for epoch in range(10):
    for x_val, y_val in zip(x_data, y_data):
        y_pred = forward(x_val) # 1) Forward pass
        l = loss(y_pred, y_val) # 2) Compute loss
        l.backward() # 3) Back propagation to update weights
        #미분의 값을 계산하여 저장
        #w.grad에 저장
        print("\tgrad: ", x_val, y_val, w.grad.item())#미분값 출력
        w.data = w.data - 0.01 * w.grad.item()#미분값이 0에 가까워지게 w수

        # Manually zero the gradients after updating weights
        w.grad.data.zero_()#초기화

    print(f"Epoch: {epoch} | Loss: {l.item()}")

# After training
print("Prediction (after training)",  4, forward(4).item())#예측값
