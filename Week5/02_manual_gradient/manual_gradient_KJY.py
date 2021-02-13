# Training data
x_data=[1.0, 2.0, 3.0]
y_data=[2.0, 4.0, 6.0]

w=1.0 # a random guess : random value

# our model forward pass (y=w*x)
def forward(x):
    return x*w

# Loss function (loss=(y*w-y)*(y*w-y))
def loss(x,y):
    y_pred=forward(x)
    return (y_pred-y)*(y_pred-y)

# compute gradient (경사도 구하기)
def gradient(x,y): # d_loss/d_w
    return 2*w*(x*w-y)

# Before Training
print("Prediction (before training)", 4, forward(4)) # 훈련하기 전 x값과 해당 x값에 의한 forward 함수 리턴값 

# Training loop
for epoch in range(10):# 10번 만큼 경사 하강법을 반복
    for x_val, y_val in zip(x_data, y_data): # zip 함수 이용하여 1:1 매칭
        # Compute derivative w,r,t to the learned weights
        # Update the weights
        # Compute the loss and print progress
        
        grad=gradient(x_val,y_val) # x_val과 y_val에 의한 경사도 구함
        w=w-0.01*grad # 경사도(grad)를 완만하게 만들게 해주는 w값으로 수정 (경사도의 절댓값을 낮춰줌)
        print("\tgrad: ", x_val, y_val) # 훈련을 위해 투입한 x_val과 y_val 출력
        l=loss(x_val, y_val) # loss 함수의 리턴값 (오차) 저장
    print("progress: ", epoch, "w=", round(w,2), "loss=", round(l,2)) # 몇 번째 경사 하강법 인지 출력
                                                                      # w값을 소수점 아래 둘째자리까지 반올림
                                                                      # 오차값을 소수점 아래 둘째자리까지 반올림

# After training
print("Predicted score (after training)", "4 hours of studying: ",forward(4)) # 훈련 후 변화된 w값을 적용한 forward 함수 탄생
                                                                                # x값(4)에 의한 forward 함수 리턴값 (예상값) 출력
