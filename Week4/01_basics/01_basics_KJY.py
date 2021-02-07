import numpy as np # numpy 모듈 가져옴 (numpy 는 np로 축약)
import matplotlib.pyplot as plt # matplotlib 모듈 가져옴 : 그래프 표현 가능 (plt로 축약)

x_data = [1.0, 2.0, 3.0] # x좌표
y_data = [2.0, 4.0, 6.0] # y좌표


# our model for the forward pass
def forward(x): # forward 함수 : x에 w를 곱해줌
    return x * w


# Loss function
def loss(x, y): # loss 함수 
    y_pred = forward(x) # 예상되는 y값 = loss 함수에서 받은 x를 forward 함수에 넣은 결과값 
    return (y_pred - y) * (y_pred - y) # (예상되는 y값 - loss 함수에서 받은 y값)**2

# List of weights/Mean square Error (Mse) for each input
w_list = [] # w 가 저장될 리스트 생성
mse_list = [] # loss 함수 결과값의 평균이 저장될 리스트 생성

for w in np.arange(0.0, 4.1, 0.1): # w 값을 0.0 부터 4.0 까지 0.1 씩 증가하는 수열을 배열로 만들어 작은 값부터 경우 따짐
    # Print the weights and initialize the lost
    print("w=", w) # w 값 출력
    l_sum = 0 # loss 함수 결과값의 합이 저장될 변수

    for x_val, y_val in zip(x_data, y_data):# zip 은 파이썬 내장함수로 두 개의 리스트를 묶어준다. 
                                            # x_val, y_val에 각각 x_data리스트와 y_data리스트의 n번째값이 들어간다.
        # For each input and output, calculate y_hat
        # Compute the total loss and add to the total error
        y_pred_val = forward(x_val) # y의 예상값
        l = loss(x_val, y_val) # loss함수에 x_val, y_val 넣었을 때 리턴값 저장
        l_sum += l # l값 더해줌
        print("\t", x_val, y_val, y_pred_val, l) 
    # Now compute the Mean squared error (mse) of each
    # Aggregate the weight/mse from this run
    print("MSE=", l_sum / len(x_data))
    w_list.append(w) # w값 w_list에 추가
    mse_list.append(l_sum / len(x_data)) # w값에서 l의 평균값 저장

# Plot it all
plt.plot(w_list, mse_list) # x 축에 w_list의 값들, y 축에 mse_list 의 값들 표시
plt.ylabel('Loss') # y 축 이름 : Loss
plt.xlabel('w') # x 축 이름 : w
plt.show() # 그래프 출력
