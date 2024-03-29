from google.colab import drive
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping # 로스값의 변화가 없을경우 빠르게 테스트 종료
from keras.layers import LSTM

drive.mount('/content/drive')
df_price = pd.read_csv('/content/drive/MyDrive/reverse.csv',encoding='utf8')   #파일을 위아래를 반전시킨후 사용함



scaler = MinMaxScaler()           # 정규화
scale_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
df_scaled = scaler.fit_transform(df_price[scale_cols]) #fit 메서드로 데이터의 분포를 추정하고, transform 메서드로 실제 데이터의 스케일을 조정합니다.

df_scaled = pd.DataFrame(df_scaled)
df_scaled.columns = scale_cols

TEST_SIZE=512 #(70퍼)

train = df_scaled[:TEST_SIZE] #앞에서 512개
test = df_scaled[TEST_SIZE:]  #앞에서 512개 제외한 나머지

def make_dataset(data, label, window_size=14):        #14개 씩 묶어서 feature 반환 그 다음 종가를 label로 반환
    feature_list = []
    label_list = []
    for i in range(len(data) - window_size):
        feature_list.append(np.array(data.iloc[i:i+window_size]))  
        label_list.append(np.array(label.iloc[i+window_size]))
    return np.array(feature_list), np.array(label_list)
   

feature_cols = ['Open', 'High', 'Low', 'Volume'] #feature 은 트레인 데이터들
label_cols = ['Close']                            #label 결과값인 종가

train_feature = train[feature_cols]
train_label = train[label_cols]  #train feature , label로 분류
test_feature = test[feature_cols] 
test_label = test[label_cols]   #test feature , label로 분류

# train dataset
train_feature, train_label = make_dataset(train_feature, train_label, 14) 

x_train= train_feature
y_train= train_label

# x_train.shape , y_train.shape
# ((498, 14, 4), (498, 1))

# test dataset (실제 예측 해볼 데이터)
test_feature, test_label = make_dataset(test_feature, test_label, 14)

# test_feature.shape, test_label.shape
# ((206, 14, 4), (206, 1)) 
model = Sequential()   #레이어를 선형으로 구성하여 연결시켜준다
model.add(LSTM(20, 
               input_shape=(14, 4),# (train_feature.shape[1], train_feature.shape[2]) 
               activation=('relu')
              )) 
model.add(Dense(1)) # 1개의 노드 output = 1
model.compile(loss='mean_squared_error', optimizer='adam')  #mean_squared_error 평균제곱오차를 계산한다. 입력 간 거리 제곱을 계산한 후 마지막 차원에 대한 평균 값이 반환된다.
                                                            #Adam( Adaptive Moment Estimation)은  각 파라미터마다 다른 크기의 업데이트를 적용하는 방법이다.            
early_stop = EarlyStopping(monitor='loss', patience=3,verbose=1) #loss가 3번까지 좋아지지않으면 멈춤

model.fit(x_train, y_train, 
                    epochs=200, 
                    batch_size=8,
                    callbacks=[early_stop])  #학습 시작

pred = model.predict(test_feature) #

plt.figure(figsize=(12, 9))
plt.plot(test_label, label='actual')
plt.plot(pred, label='prediction')
plt.legend()
plt.show()

#참고 https://teddylee777.github.io/tensorflow/LSTM%EC%9C%BC%EB%A1%9C-%EC%98%88%EC%B8%A1%ED%95%B4%EB%B3%B4%EB%8A%94-%EC%82%BC%EC%84%B1%EC%A0%84%EC%9E%90-%EC%A3%BC%EA%B0%80
#https://user-images.githubusercontent.com/77331618/119235798-2c10e200-bb6f-11eb-83f2-761952ad3765.png












Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
Epoch 1/200
63/63 [==============================] - 1s 4ms/step - loss: 0.1703
Epoch 2/200
63/63 [==============================] - 0s 5ms/step - loss: 0.0320
Epoch 3/200
63/63 [==============================] - 0s 5ms/step - loss: 0.0034
Epoch 4/200
63/63 [==============================] - 0s 5ms/step - loss: 0.0025
Epoch 5/200
63/63 [==============================] - 0s 5ms/step - loss: 0.0030
Epoch 6/200
63/63 [==============================] - 0s 5ms/step - loss: 0.0025
Epoch 7/200
63/63 [==============================] - 0s 5ms/step - loss: 0.0020
Epoch 8/200
63/63 [==============================] - 0s 5ms/step - loss: 0.0026
Epoch 9/200
63/63 [==============================] - 0s 5ms/step - loss: 0.0022
Epoch 10/200
63/63 [==============================] - 0s 6ms/step - loss: 0.0020
Epoch 11/200
63/63 [==============================] - 0s 5ms/step - loss: 0.0019
Epoch 12/200
63/63 [==============================] - 0s 5ms/step - loss: 0.0016
Epoch 13/200
63/63 [==============================] - 0s 5ms/step - loss: 0.0020
Epoch 14/200
63/63 [==============================] - 0s 5ms/step - loss: 0.0022
Epoch 15/200
63/63 [==============================] - 0s 5ms/step - loss: 0.0020
Epoch 16/200
63/63 [==============================] - 0s 5ms/step - loss: 0.0016
Epoch 17/200
63/63 [==============================] - 0s 5ms/step - loss: 0.0015
Epoch 18/200
63/63 [==============================] - 0s 5ms/step - loss: 0.0022
Epoch 19/200
63/63 [==============================] - 0s 5ms/step - loss: 0.0015
Epoch 20/200
63/63 [==============================] - 0s 5ms/step - loss: 0.0015
Epoch 21/200
63/63 [==============================] - 0s 5ms/step - loss: 0.0013
Epoch 22/200
63/63 [==============================] - 0s 5ms/step - loss: 0.0015
Epoch 23/200
63/63 [==============================] - 0s 5ms/step - loss: 0.0013
Epoch 24/200
63/63 [==============================] - 0s 5ms/step - loss: 0.0014
Epoch 25/200
63/63 [==============================] - 0s 5ms/step - loss: 0.0015
Epoch 26/200
63/63 [==============================] - 0s 5ms/step - loss: 9.3433e-04
Epoch 27/200
63/63 [==============================] - 0s 5ms/step - loss: 0.0015
Epoch 28/200
63/63 [==============================] - 0s 5ms/step - loss: 0.0012
Epoch 29/200
63/63 [==============================] - 0s 5ms/step - loss: 0.0012
Epoch 00029: early stopping


