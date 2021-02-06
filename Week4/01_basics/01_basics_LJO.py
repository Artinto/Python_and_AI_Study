import numpy as np #numpy를 np로 호출
import matplotlib.pyplot as plt #그래프 모듈을 plt로 호출

x_data = [1.0, 2.0, 3.0]  #x,y의 입력데이터
y_data = [2.0, 4.0, 6.0]


# our model for the forward pass
def forward(x): #선형모델, x와 random value인 w(기울기)fmf 곱하는 함수
    return x * w


# Loss function
def loss(x, y): #loss값을 구하는 함수, forward함수의 리턴값을 받아와서 입력데이터인y와의 차를 구한후 제곱(차가 +,-가 되기때문에 제곱)
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)

# List of weights/Mean square Error (Mse) for each input
w_list = [] #w값의 리스트
mse_list = [] #임의의 w값에 대하여 계산된 loss함수의 평균값 리스트

for w in np.arange(0.0, 4.1, 0.1): #w의 범위는 0.0~4.0, 0.1씩 증가
    # Print the weights and initialize the lost
    print("w=", w)
    l_sum = 0

    for x_val, y_val in zip(x_data, y_data): #x_val는 x_data의 값, y_val=y_data의 값
        # For each input and output, calculate y_hat
        # Compute the total loss and add to the total error
        y_pred_val = forward(x_val) #x_val와 w를 곱한값
        l = loss(x_val, y_val) #forward함수의 값과 y_val의 차의 제곱
        l_sum += l #loss값의 합
        print("\t", x_val, y_val, y_pred_val, l)
    # Now compute the Mean squared error (mse) of each
    # Aggregate the weight/mse from this run
    print("MSE=", l_sum / len(x_data)) #mse출력
    w_list.append(w) #w리스트에 w값 삽입
    mse_list.append(l_sum / len(x_data)) #mse리스트에 mse값 삽입

# Plot it all
plt.plot(w_list, mse_list) #x축에 w_list의 값, y축에 mse_list의 값 적용
plt.ylabel('Loss') #y축 이름은 Loss
plt.xlabel('w') #x축 이름은 
plt.show()
