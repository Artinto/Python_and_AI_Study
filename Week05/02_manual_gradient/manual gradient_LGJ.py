# Training Data
x_data = [1.0, 2.0, 3.0]    # x데이터 입력
y_data = [2.0, 4.0, 6.0]    # y데이터 입력

w = 1.0  # a random guess: random value, w랜덤값 입력


# our model forward pass
def forward(x): # foward함수를 통해 x*w 계산
    return x * w    # y^ 


# Loss function
def loss(x, y): # 손실함수를 통해 예측값과 원값의 오차의 제곱 계산
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


# compute gradient
def gradient(x, y):  # d_loss/d_w,  (y_pred - y)^2를 직접 미분해서 함수를 만들고 기울기를 계산
    return 2 * x * (x * w - y)


# Before training
print("Prediction (before training)",  4, forward(4))   # 기계 학습 전 4를 대입했을 때의 forward 값: 4 x 1.0

# Training loop
for epoch in range(10): # 10번 경사 하강
    for x_val, y_val in zip(x_data, y_data):    # x_val, y_val에 각각 x_data, y_data의 쌍들을 집어넣음
        # Compute derivative w.r.t to the learned weights
        # Update the weights
        # Compute the loss and print progress
        
        grad = gradient(x_val, y_val)   # x_val ,y_val를 gradient 함수에 넣고 그때의 기울기를 계산하여 grad에 대입
        w = w - 0.01 * grad     # 경사 하강법 실행, 학습률은 0.01
        print("\tgrad: ", x_val, y_val, round(grad, 2)) # x_val, y_val, grad를 소숫점 둘째까지 반올림 하여 출력
        l = loss(x_val, y_val)  # 변수 l에 x_val, y_val, 그리고 경사하강법 후 최적의 w에 따른 loss함수 결괏값을 대입
    print("progress:", epoch, "w=", round(w, 2), "loss=", round(l, 2))  # 경사하강법의 반복횟수, w(소숫점 2자리반올림), 오차함숫값 l(소숫점 2자리반올림) 출력

# After training
print("Predicted score (after training)",  "4 hours of studying: ", forward(4)) # 기계 학습 후 4를 대입했을 때의 값
