# Training Data
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0  # a random guess: random value
#임의의 수


# our model forward pass
def forward(x): 
    return x * w
#영상에서 나온 y^


# Loss function
def loss(x, y): #오차 계산 
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


# compute gradient
def gradient(x, y):  # d_loss/d_w
    return 2 * x * (x * w - y) #기울기 구하기 


# Before training
print("Prediction (before training)",  4, forward(4))

# Training loop
for epoch in range(10): #10번 실행 
    for x_val, y_val in zip(x_data, y_data): 
        # Compute derivative w.r.t to the learned weights
        # Update the weights
        # Compute the loss and print progress
        
        grad = gradient(x_val, y_val) #
        w = w - 0.01 * grad # 임의의값(w)을 변경하며 최적의 w를 찾아나가는 작업 
        print("\tgrad: ", x_val, y_val, round(grad, 2))#round= 반올림 2번째 자리까지 
        l = loss(x_val, y_val) #
    print("progress:", epoch, "w=", round(w, 2), "loss=", round(l, 2))

# After training
print("Predicted score (after training)",  "4 hours of studying: ", forward(4))
