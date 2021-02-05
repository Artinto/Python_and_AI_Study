# numpy 모듈 가져오기
# 데이터를 차트나 plot으로 그려주는 모듈 가져오기
import numpy as np
import matplotlib.pyplot as plt

# 선형 모델에 들어갈 input값 x데이터와 결과값인 y데이터 지정
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


# our model for the forward pass
# forward 함수를 통해 x가 들어오면 x(input 값) * w(가중치)값을 반환
def forward(x):
    return x * w

# Loss function
# 손실함수로 x와 y가 들어오면 실행됨
# y_pred는 forward(단순화한 선형 모델 함수)함수를 통해 반환한 값을 가짐
# 손실함수는 x에 값에 따라 예측된 y값과 훈련셋의 y값과의 차이를 제곱한 값을 반환함(양수여야 하기 때문)  
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)

# List of weights/Mean square Error (Mse) for each input
# 가중치 리스트와 평균제곱오차 리스트를 만듦
w_list = []
mse_list = []

# for문을 이용하여 가중치 w값을 0.0부터 4.1(포함X)까지0.1만큼 더해진 값으로 대입
# 가중치 값을 출력하고 손실 합산 값을 저장하는 변수를 0으로 초기화
for w in np.arange(0.0, 4.1, 0.1):
    # Print the weights and initialize the lost
    print("w=", w)
    l_sum = 0

    # 내장 함수 zip은 동일한 개수로 이루어진 자료형을 묶어주는 역할을 하는 함수로
       x_val과 y_val에 각각 x_data와 y_data 값을 대입
    # forward함수를 통해 x_val값으로 단순 선형 모델 식으로 계산후, y_pred_val에 대입
    # 손실함수를 통해 나온 값을 l에 저장
    # for문이 끝날 때까지 오차값을 l_sum에 더함
    # x_val, y_val, y_pred_val, l값을 출력
    for x_val, y_val in zip(x_data, y_data):
        # For each input and output, calculate y_hat
        # Compute the total loss and add to the total error
        y_pred_val = forward(x_val)
        l = loss(x_val, y_val)
        l_sum += l
        print("\t", x_val, y_val, y_pred_val, l)
    # Now compute the Mean squared error (mse) of each
    # Aggregate the weight/mse from this run
    # x_data길이만큼(길이:3) l_sum값을 나눔(=평균제곱오차값) 그리고 출력
    # 가중치 리스트에 가중치 값을 추가
    # mse 리스트에 평균제곱오차값을 추가
    print("MSE=", l_sum / len(x_data))
    w_list.append(w)
    mse_list.append(l_sum / len(x_data))

# Plot it all
# plot x축은 가중치 리스트 값으로, y축은 평균제곱오차 리스트 값으로 설정
# y축 레이블을 Loss로, x축 레이블을 w로 지정
# plot을 화면에 보여줌
plt.plot(w_list, mse_list)
plt.ylabel('Loss')
plt.xlabel('w')
plt.show()
