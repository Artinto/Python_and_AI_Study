import numpy as np              # numpy import
import matplotlib.pyplot as plt # matplotlib.pyplot import (데이터 차트나 플롯 그리기용)

x_data = [1.0, 2.0, 3.0] # input  data
y_data = [2.0, 4.0, 6.0] # output data (x_data에 mapping 되는 값)


# our model for the forward pass
def forward(x):  # 선형모델 계산
    return x * w # W == weight (가중치)


# Loss function (손실 함수)
# 우리가 세운 가설과 얼마나 차이가 있는지 계산
def loss(x, y): 
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y) # 차이의 제곱 return (cuz, 음수가 나올수도 있어서)

# List of weights/Mean square Error (Mse) for each input
w_list = []   # weight list
mse_list = [] # cost list

# weight의 변화에 따른 cost 구하기 => 학습
# for(auto w = 0.0; w < 4.1; w += 0.1)
for w in np.arange(0.0, 4.1, 0.1):
    # Print the weights and initialize the lost
    print("w=", w) # print weight
    l_sum = 0      # sum of loss

    # 각각의 input, output data의 변화에 따른 loss의 합 계산
    # zip   : 동일한 개수로 이루어진 자료형을 묶어 주는 역할을 하는 함수
    # x_val : x_data의 value
    # y_val : y_data의 value
    for x_val, y_val in zip(x_data, y_data):
        # For each input and output, calculate y_hat
        # Compute the total loss and add to the total error
        # y_pred_val : 선형모델 계산 값
        # l          : loss 값
        y_pred_val = forward(x_val)
        l = loss(x_val, y_val)
        l_sum += l
        print("\t", x_val, y_val, y_pred_val, l)
    # Now compute the Mean squared error (mse) of each
    # Aggregate the weight/mse from this run
    # l_sum / len(x_data) == cost
    print("MSE=", l_sum / len(x_data))
    w_list.append(w)
    mse_list.append(l_sum / len(x_data))

# Plot it all
plt.plot(w_list, mse_list) # x축 : w_list  y축 : mse_list
plt.ylabel('Loss')         # y축 이름 : Loss
plt.xlabel('w')            # x축 이름 : w
plt.show()                 # draw graph