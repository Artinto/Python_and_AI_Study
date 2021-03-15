plt.show()
# Training Data #x,y 데이터값 입력
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0  # a random guess: random value


# our model forward pass #선형함수 설정
def forward(x):
    return x * w


# Loss function     #loss 구하기
def loss(x, y): 
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


# compute gradient #기울기를 구하는 것으로 loss의 변화량을 w의 변화량으로 나눈 식
def gradient(x, y):  # d_loss/d_w
    return 2 * x * (x * w - y)


# Before training #
print("Prediction (before training)",  4, forward(4))
#4를 넣었을때에 리턴값을 출력
# Training loop 
for epoch in range(10): #gradient를 10번 실행
    for x_val, y_val in zip(x_data, y_data):
        # Compute derivative w.r.t to the learned weights
        # Update the weights
        # Compute the loss and print progress
        
        grad = gradient(x_val, y_val) #기울기값  저장
        w = w - 0.01 * grad         # 알맞은 값을 찾기위해 w값을 조근씩 줄여줌
        print("\tgrad: ", x_val, y_val, round(grad, 2)) #소수점 아래 2번째까지만 출력 그밑으론 반홀림
        l = loss(x_val, y_val)  #loss 값 저장
    print("progress:", epoch, "w=", round(w, 2), "loss=", round(l, 2))

# After training
print("Predicted score (after training)",  "4 hours of studying: ", forward(4))
#다시 4를 넣어 값을 확인
