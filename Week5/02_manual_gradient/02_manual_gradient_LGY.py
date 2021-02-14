# 데이터 선언
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# initial weight
w = 1.0  # random한 임의의 1.0에서 weight시작


# model : Linear
def forward(x):
    return x * w


# Loss function
def loss(x, y):
    y_pred = forward(x) # y헷
    return (y_pred - y) * (y_pred - y) #(예측 - 실제)^2


# gradient 계산
def gradient(x, y):  
    # loss function의 w편미분
    # loss(w) = (x*w - y)^2 
    return 2 * x * (x * w - y)


# 학습이 하나도 되지 않았을 때의 예측값
print("Prediction (before training)",  4, forward(4)) # 모델에 x = 4가 들어갔을 때 return값(w*x) //(w : 초기값 0.1)

# training 과정
for epoch in range(10): # 10번 반복학습시키겠다
    for x_val, y_val in zip(x_data, y_data):
        grad = gradient(x_val, y_val) # loss function의 w편미분한 gradient구하기
        w = w - 0.01 * grad # 구한 gradient값으로 weight updata하기
        print("\tgrad: ", x_val, y_val, round(grad, 2)) # '\t' : tab, round(grad, 2) : grad값을 소수점 둘째자리까지 나타내기
        l = loss(x_val, y_val) # loss func을 통해 loss값 구하기
    print("progress:", epoch, "w=", round(w, 2), "loss=", round(l, 2))

# 학습이 끝난 후 모델의 예측값
print("Predicted score (after training)",  "4 hours of studying: ", forward(4)) # 모델에 x = 4가 들어갔을 때 return값(w*x) //(w : 학습된 w)
