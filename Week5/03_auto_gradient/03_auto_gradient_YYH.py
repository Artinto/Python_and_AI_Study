import torch
import pdb

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
w = torch.tensor([1.0], requires_grad=True) #연산 기록
# our model forward pass
def forward(x):
    return x * w    #선형모델설정
# Loss function
def loss(y_pred, y_val):
    return (y_pred - y_val) ** 2    #loss함수 설정

# Before training
print("Prediction (before training)",  4, forward(4).item()) 
#.item() tensor에 하나의 값만 존재할경우 출력
#학습 전 출력값.


# Training loop
for epoch in range(10):     #10번 학습
    for x_val, y_val in zip(x_data, y_data):
        y_pred = forward(x_val) # 1) Forward pass 
        l = loss(y_pred, y_val) # 2) Compute loss
        l.backward() # 3) Back propagation을 통해 w값 조정
                        #
        print("\tgrad: ", x_val, y_val, w.grad.item()) # x,y, 미분값을 출력
        w.data = w.data - 0.01 * w.grad.item() #grad가 완화되도록 w.data 값을 조정

        # Manually zero the gradients after updating weights
        w.grad.data.zero_() #w.grad 데이터 초기화

    print(f"Epoch: {epoch} | Loss: {l.item()}") # f-strings , 반복횟수와 loss를 출력
            
# After trainin 학습 후 출력값
print("Prediction (after training)",  4, forward(4).item())     
