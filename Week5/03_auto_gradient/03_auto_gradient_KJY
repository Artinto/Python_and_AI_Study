import torch # tensor, grad 등의 함수를 가진 모듈
import pdb # 파이썬을 위한 대화형 디버거를 제공하는 모듈

x_data=[1.0,2.0,3.0]
y_data=[2.0,4.0,6.0]
w=torch.tensor([1.0], requires_grad=True) 
# tensor 을 통해 한가지 data type 을 갖는 요소들의 행렬을 만듦
# requires_grad=True -> 해당 변수는 학습을 통해 변경 가능

# our model forward pass (y=w*x)
def forward(x):
    return x*w

# Loss function (loss=(y*w-y)*(y*w-y))
def loss(y_pred, y_val):
    return (y_pred-y_val)**2

# Before training
print("Prediction (before training)", 4, forward(4).item())

# Training loop
for epoch in range(10):
    for x_val, y_val in zip(x_data, y_data):
        y_pred=forward(x_val) # Forward 함수에 x_val을 pass
        l=loss(y_pred,y_val) # loss 함수 결과값 l에 저장
        l.backward() # Back propagation(역전파) 사용하여 weights 값 수정
        print("\tgrad: ", x_val, y_val, w.grad.item()) # x,y값에 따른 경사도 값 출력 (grad 는 경사도 구하는 파이토치 내장함수)
        w.data = w.data - 0.01 * w.grad.item() # 경사도(grad)를 완만하게 만들게 해주는 w값으로 수정 (경사도의 절댓값을 낮춰줌)
        
        # Manually zero gradients after updating weights
        w.grad.data.zero_() # w변수 수정 후 수동으로 grad 값을 0으로 만듦
    
    print(f"Epoch: {epoch} | Loss: {l.item()}") # f 문자열 포매팅 이용하여 epoch 값과 loss 값 출력

# After training
print("Prediction (after training)", 4, forward(4).item()) # x=4 일 때, 지금까지의 훈련에 의한 y예측값 출력
