x_data = [1.0, 2.0, 3.0] #주어진 x,y의 데이터셋
y_data = [2.0, 4.0, 6.0]

w = 1.0  # 주어진 기울기값

# our model forward pass
def forward(x): #일차함수식 계산
    return x * w

# Loss function
def loss(x, y): #forward함수의 값을 받아 원래의 값과의 차이의 제곱(오차)
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)

# compute gradient
def gradient(x, y):  # d_loss/d_w #2차함수인 loss의 기울기 계산
    return 2 * x * (x * w - y)

# Before training
print("Prediction (before training)",  4, forward(4))

# Training loop
for epoch in range(10): #10번 학습
    for x_val, y_val in zip(x_data, y_data): #주어진 데이터 셋에 대하여 계산
        # Compute derivative w.r.t to the learned weights
        # Update the weights
        # Compute the loss and print progress
        
        grad = gradient(x_val, y_val) #기울기 계산
        w = w - 0.01 * grad #w값을 변화시키며 최저점의 w값을 찾아나감, 0.01은 learning rate으로 적절한 값이 주어져야 함
        print("\tgrad: ", x_val, y_val, round(grad, 2))
        l = loss(x_val, y_val) #loss함수의 값 저장
    print("progress:", epoch, "w=", round(w, 2), "loss=", round(l, 2))

# After training
print("Predicted score (after training)",  "4 hours of studying: ", forward(4))
