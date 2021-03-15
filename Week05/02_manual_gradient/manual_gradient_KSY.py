# Training Data
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0  # a random guess: random value (어떤값을 넣어도 상관없다 시작하는 지점)


# our model forward pass y=xw라는 기본식
def forward(x):
    return x * w


# Loss function    ;; 저번주에 사용했던 loss=(y_pred-y)**2
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


# compute gradient  수학적의미로 기울기 loss의 변화량에서 w의 변화량을 나눈값
def gradient(x, y):  # d_loss/d_w
    return 2 * x * (x * w - y)


# Before training
print("Prediction (before training)",  4, forward(4))#설정한 w값에 4를 넣은 리턴값

# Training loop
for epoch in range(10):# 10번만큼  gradient를 실행하는데 횟수가 많을수록 알맞은 w값에 가까워짐
    for x_val, y_val in zip(x_data, y_data):  #((x1,y1),(x2,y2),(x3,y3))
        # Compute derivative w.r.t to the learned weights
        # Update the weights
        # Compute the loss and print progress
        grad = gradient(x_val, y_val) #x ,y값 함수에 대입
        w = w - 0.01 * grad #계산된 기울기 조금씩 w에 값을 바꿔주면서 알맞은  w를 찾아냄 여기서 소수점이 많아질수록 기울기의변화가 더 적어져서 많은 loop를 돌려야 값이 나오지만 더 정확하게나옴
        print("\tgrad: ", x_val, y_val, round(grad, 2))#소수점 아래 2번째까지 반올림
        l = loss(x_val, y_val) #loss함수값 저장
    print("progress:", epoch, "w=", round(w, 2), "loss=", round(l, 2)) #몇번째 loop인지, w값은 몇인지, loss값은 몇인지 나타냄

# After training
print("Predicted score (after training)",  "4 hours of studying: ", forward(4)) #모든  loop가 끝난뒤 다시4를 넣어서 나온 값을 보고  x,y의 관계식을 알아낸 것을 확인

