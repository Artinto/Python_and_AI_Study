# Training Data
# input 데이터인 x data와 결과값인 y data를 리스트에 저장
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# weight값으로 무작위 값인 1.0을 대입
w = 1.0  # a random guess: random value


# our model forward pass
# 인자 x를 받는 순방향 진행 함수로 선형 함수인 x * w를 반환
def forward(x):
    return x * w


# Loss function
# 손실함수로 인자 x와 y를 받으며 선형 모델 함수를 통해 나온 값을 y_pred에 대입(선형 모델 식을 통해 나온 예측 y값)
# 손실함수는 x에 값에 따라 예측된 y값과 훈련셋의 y값과의 차이를 제곱한 값을 반환함(양수여야 하기 때문) 
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


# compute gradient
# 손실함수의 기울기를 계산하는 함수로 loss값을 w에 대해 미분한 값을 반환함. ((xw-y)^2값을 미분하면 2x(xw-y)가 나옴))
def gradient(x, y):  # d_loss/d_w
    return 2 * x * (x * w - y)


# Before training
# 훈련 전의 예측값은 input값인 x에 4를 대입했을 때 결과값으로 4가 나온다는 것을 출력(w값이 1.0이므로)
print("Prediction (before training)",  4, forward(4))

# Training loop
# 반복 훈련 과정으로 for문을 이용하여 10번(0~9) 반복함
# x_val와 y_val에 x_data와 y_data를 차례로 대입함( zip 함수는 동일한 개수로 이루어진 자료형을 묶어 주는 역할을 하는 함수)
for epoch in range(10):
    for x_val, y_val in zip(x_data, y_data):
        # Compute derivative w.r.t to the learned weights
        # Update the weights
        # Compute the loss and print progress
        # 기울기를 계산하는 함수 gradient에 x_val와 y_val값을 인자로 넘겨 반환된 값인 손실함수의 기울기 값을 grad에 저장
        # weight 값인 w값에 loss함수를 미분해서 얻은 기울기에 alpha값인 0.01을 곱한 값을 빼준 후 그 값을 w에 대입하여 weight값을 업데이트 시킴
        # alpha는 학습률로 학습속도를 조절하는 상수/w에서 0.01 * grad를 빼준 이유는 기울기가 양수라면 w값이 더 작아야 하고 기울기가 음수라면 w값을 더 크게 만들어야 하기 때문
        # x_val, y_val값과 loss함수 기울기값인 grad를 round함수를 이용해 반올림하여 소수 둘째자리까지 나타내어 출력
        # x_val와 y_val를 loss함수의 인자로 넘겨 반환된 값(손실함수 값)을 l에 저장
        # 훈련 반복 횟수인 epoch 값을 출력하고 w값과 loss 값을 반올림하여 소수 둘째자리 까지 나타내어 출력
        grad = gradient(x_val, y_val)
        w = w - 0.01 * grad
        print("\tgrad: ", x_val, y_val, round(grad, 2))
        l = loss(x_val, y_val)
    print("progress:", epoch, "w=", round(w, 2), "loss=", round(l, 2))

# After training
# 훈련 후에 input 값을 4로 했을 때 결과값을 출력
print("Predicted score (after training)",  "4 hours of studying: ", forward(4))
