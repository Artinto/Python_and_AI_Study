import numpy as np
import matplotlib.pyplot as plt#데이터를 차트로 그려주는 모듈

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


def forward(x): #가중치를 나타내주는 함수
    return x * w #w는 임의의값으로 설정 



def loss(x, y): # 오차값 
    y_pred = forward(x) # 영상에서의 y^
    return (y_pred - y) * (y_pred - y)# 오차는 원하는 값에서 임의의값까지의 차이이기에 제곱을 통하여 계산 


w_list = []    # 가중치의 리스트
mse_list = []  #MSE의 리스트

# 여기서부터 반복구간
for w in np.arange(0.0, 4.1, 0.1): # 0.0부터 4.0까지 0.1단위로 가중치를 넣어준다.
    # Print the weights and initialize the lost
    print("w=", w)
    l_sum = 0

    for x_val, y_val in zip(x_data, y_data): #(x,y)좌표형식 
       
        
        y_pred_val = forward(x_val) # y^ 구하는 식
        l = loss(x_val, y_val) #오차 구하기
        l_sum += #오차들의 합
        print("\t", x_val, y_val, y_pred_val, l)
    print("MSE=", l_sum / len(x_data)) #MSE란 loss들의 모든 합을 loss의 갯수만큼 나눈 수
    w_list.append(w) #가중치의 리스트에 가중치들 삽입
    mse_list.append(l_sum / len(x_data)) # MSE리스트에 MSE삽입

    #여기까지 w의 range만큼 반복

# Plot it all
plt.plot(w_list, mse_list) 
plt.ylabel('Loss') #y축 라벨
plt.xlabel('w') #x축 라벨
plt.show() # 그래프

