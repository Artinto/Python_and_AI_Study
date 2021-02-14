ta = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0  # a random guess: random value


# our model forward pass
def forward(x):#선형함수 선언 y^구하기
    return x * w


# Loss function
def loss(x, y):#예측값과 실제값의 차ㅣ
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


# compute gradient
def gradient(x, y):  # d_loss/d_w#loss의 미분값 구하기
    return 2 * x * (x * w - y)


# Before training
print("Prediction (before training)",  4, forward(4))#x가 4일때 

# Training loop
for epoch in range(10):#10번 반복
    for x_val, y_val in zip(x_data, y_data):
        # Compute derivative w.r.t to the learned weights
        # Update the weights
        # Compute the loss and print progress
        
        grad = gradient(x_val, y_val)#미분값 구하기
        w = w - 0.01 * grad#미분값을 이용하여 w값을 구하기
        print("\tgrad: ", x_val, y_val, round(grad, 2))#grad값을 소수점 2번째 까지
        l = loss(x_val, y_val)
    print("progress:", epoch, "w=", round(w, 2), "loss=", round(l, 2))

# After training
print("Predicted score (after training)",  "4 hours of studying: ", forward(4))#결과
