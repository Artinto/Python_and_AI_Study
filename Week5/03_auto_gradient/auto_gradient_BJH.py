import torch
import pdb

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0] # 고정된  x,y
w = torch.tensor([1.0], requires_grad=True) #tensor는 행

# our model forward pass
def forward(x): 
    return x * w

# Loss function
def loss(y_pred, y_val): # 오차 계산 
    return (y_pred - y_val) ** 2

# Before training
print("Prediction (before training)",  4, forward(4).item()) # x에 4대입해서 forward 계

# Training loop
for epoch in range(10): #10번 실행 
    for x_val, y_val in zip(x_data, y_data):
        y_pred = forward(x_val) # y^ 계산  
        l = loss(y_pred, y_val) # loss 저장
        l.backward() # 3) w의 값을 조정 
        print("\tgrad: ", x_val, y_val, w.grad.item()) # x,y,미분값 나타내기 
        w.data = w.data - 0.01 * w.grad.item() # grad를 최적화하기 위한 w.data 조정 

        # Manually zero the gradients after updating weights
        w.grad.data.zero_() # w.grad 초기화 

    print(f"Epoch: {epoch} | Loss: {l.item()}")

# After training
print("Prediction (after training)",  4, forward(4).item())
