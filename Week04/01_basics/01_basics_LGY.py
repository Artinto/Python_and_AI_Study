#01_basics.py

import numpy as np  # 행렬계산을 용이하게 해주는 라이브러리
import matplotlib.pyplot as plt  # 시각적으로 볼 수 있도록 그래프를 만들어주는 라이브러리

x_data = [1.0, 2.0, 3.0]    # 학습시킬 문제 x data
y_data = [2.0, 4.0, 6.0]    # 학습시킬 답안 y data


# 함수실행시 실행되는 함수
# linear regression
# y_pred_val = forward(x_val)  // line 실행시 x_val이 통과하는 forward함수
def forward(x):
    return x * w



# l = loss(x_val, y_val)  // line 실행시 x_val, y_val가 통과하는 loss함수
def loss(x, y): # Loss function
    y_pred = forward(x)   #  forward(x_val) 실행
    # y_pred = x_val * w
    return (y_pred - y) * (y_pred - y)  # (x_val * w - y_val)^2






w_list = [] 
mse_list = []

for w in np.arange(0.0, 4.1, 0.1):  # Weight 값 : 0.0에서 4.1전까지  0.1씩 증가한 array만들기
    # Print the weights and initialize the lost
    print("w=", w)
    l_sum = 0 #loss값들의 합 : x_data*w(예측값)과 y_data(실제값)과의 오차들의 합

    for x_val, y_val in zip(x_data, y_data): # 각각의 학습데이터를 가져옴
        y_pred_val = forward(x_val)  # 학습데이터 x를 forward라는 함수에 넣어줌. (forward함수 실행)
        # (return x * w) y_pred_val에는 x_val * W 값이 들어감.
        l = loss(x_val, y_val) # 두개의 데이터가 loss함수를 거침. (loss함수 실행)
        # return (y_pred - y) * (y_pred - y)  # (x_val * w - y_val)^2
        l_sum += l #loss값들의 합
        print("\t", x_val, y_val, y_pred_val, l) # /t : tab

    print("MSE=", l_sum / len(x_data)) # MSE값 print
    w_list.append(w) 
    mse_list.append(l_sum / len(x_data))
print(w_list,"asdfasdf")

plt.plot(w_list, mse_list) # x축에 w_list, y축엔 mse_list을 나타내기
plt.ylabel('Loss') # x축 이름
plt.xlabel('w') # y축 이름
plt.show() # 그래프 그려라
