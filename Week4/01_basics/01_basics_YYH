import numpy as np   #numpy 모듈 삽입
import matplotlib.pyplot as plt  # matplotlib.pyplot 모듈삽입(차트 플롯)

x_data = [1.0, 2.0, 3.0] #x_data 지정
y_data = [2.0, 4.0, 6.0] #y_data 지정


# our model for the forward pass
def forward(x):     #선형모델함수, x*w(임의의값) 를 반환
    return x * w


# Loss function 으로 예측 y값에서 y를 뺌으로 오차를 구한후 제곱
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)

# List of weights/Mean square Error (Mse) for each input
w_list = []     #가중치 리스트
mse_list = []   #오차의 제곱에 평균값 리스트


#w에 0부터 0.1씩 증가시켜 4까지 for문
for w in np.arange(0.0, 4.1, 0.1):
    # Print the weights and initialize the lost
    print("w=", w)      #w 출력   
    l_sum = 0           #loss의 총합


#zip는 같은 길이의 리스트를 같은 인덱스끼리 잘라서 리스트로 반환해줌
    for x_val, y_val in zip(x_data, y_data):
        # For each input and output, calculate y_hat
        # Compute the total loss and add to the total error
        y_pred_val = forward(x_val)             # forward로 예측 y값 구함
        l = loss(x_val, y_val)                  # l에 loss 값 입력 
        l_sum += l                              #l_sum 값에 l 더함
        print("\t", x_val, y_val, y_pred_val, l) #출력
    # Now compute the Mean squared error (mse) of each
    # Aggregate the weight/mse from this run
    print("MSE=", l_sum / len(x_data))      # MSE값 출력
    w_list.append(w)                        #w_list에 추가
    mse_list.append(l_sum / len(x_data))    #mse_list에 추가

# Plot it all
plt.plot(w_list, mse_list) # x엔 w_list, y엔 mse_list 
plt.ylabel('Loss')      #y축 이름 LOSS
plt.xlabel('w')         #x축 이름 w
plt.show()
