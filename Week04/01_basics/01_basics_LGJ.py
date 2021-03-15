  
import numpy as np  # numpy를 np라는 입력으로 호출받도록 불러옴
import matplotlib.pyplot as plt # 그래프를 띄워주는 라이브러리인 matplotlib.pyplot를 plt라고 입력하면 호출받도록 불러옴

x_data = [1.0, 2.0, 3.0]  # x의 입력 데이터값 예시를 줌.
y_data = [2.0, 4.0, 6.0]  # y의 출력 데이터값 예시를 줌.


# our model for the forward pass
# foward라는 함수로 x를 입력받으면 x*w를 반환
def forward(x): 
    return x * w


# Loss function
 #loss라는 함수로 x,y를 입력받으면 y_pred에 x*w를 넣고, y_pred와 y의 차이의 제곱을 반환
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)

# List of weights/Mean square Error (Mse) for each input
# w와 오차의 제곱의 평균인 mse에 관한 리스트 생성
w_list = []
mse_list = []

# w를 0.0이상부터 4.1미만까지 0.1까지 증가시키고, 안에 있는 문장들을 실행
for w in np.arange(0.0, 4.1, 0.1):
    # Print the weights and initialize the lost
    print("w=", w)
    l_sum = 0

    for x_val, y_val in zip(x_data, y_data):  # for문을 활용해 x와 y의 데이터값들 안의 순서쌍을 하나씩 증가시키면서 안에 있는 문장을 실행
        # For each input and output, calculate y_hat
        # Compute the total loss and add to the total error
        y_pred_val = forward(x_val) # y_pred_val = x_val*w
        l = loss(x_val, y_val)  # y와 y 예측값의 오차를 변수 l에 입력
        l_sum += l  # l값들의 합 구하기(data 순서쌍이 3개 이므로 3개를 더함)
        print("\t", x_val, y_val, y_pred_val, l)
    # Now compute the Mean squared error (mse) of each
    # Aggregate the weight/mse from this run
    print("MSE=", l_sum / len(x_data))  # 변수 MSE에 l의 합들을 x의 개수인 3으로 나눔, 즉 오차들의 평균
    # w와 mse에 관한 함수를 만들기 위해 w_list에 w값을, mse_list에 l_sum/len(x_data)값들을 추가
    w_list.append(w)
    mse_list.append(l_sum / len(x_data))

# Plot it all
# 그래프로 띄우는 작업
plt.plot(w_list, mse_list)  #x축에 w_list값들을, y축에 mse_list값들을 대입
plt.ylabel('Loss')  #y축의 이름은 Loss
plt.xlabel('w') #x축의 이름은 w
plt.show() #그래프 
