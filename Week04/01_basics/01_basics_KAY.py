import numpy as np
import matplotlib.pyplot as plt #데이터를 시각화하는 라이브러리 

x_data = [1.0, 2.0, 3.0]#x좌표
y_data = [2.0, 4.0, 6.0]#y좌표


# our model for the forward pass
def forward(x):
    return x * w #w는 0부터 4까지 0.1단위로 


# Loss function
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)#차이의 크기

# List of weights/Mean square Error (Mse) for each input
w_list = []# w가 저장될 리스트
mse_list = []#loss함수에서 나온 결과 값의 평균 저장

for w in np.arange(0.0, 4.1, 0.1):#0.0부터 4까지 0.1단위로 반복
    # Print the weights and initialize the lost
    print("w=", w)
    l_sum = 0 #loss값을 받을 변수

    for x_val, y_val in zip(x_data, y_data):#zip은 리스트 묶어준
        # For each input and output, calculate y_hat
        # Compute the total loss and add to the total error
        y_pred_val = forward(x_val) #y예측값
        l = loss(x_val, y_val)
        l_sum += l#l값 더해줌
        print("\t", x_val, y_val, y_pred_val, l)
    # Now compute the Mean squared error (mse) of each
    # Aggregate the weight/mse from this run
    print("MSE=", l_sum / len(x_data))
    w_list.append(w)#list에 입력
    mse_list.append(l_sum / len(x_data))#list에 입력

# Plot it all
plt.plot(w_list, mse_list)#x축-w_list, y축_mse_list
plt.ylabel('Loss')#y축 이름
plt.xlabel('w')#x축이름
plt.show()#그래프
