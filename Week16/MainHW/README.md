# [실습] 주식데이터를 이용한 RNN 코드 작성


### ※ 준수사항
- .py 혹은 .ipynb 파일로 업로드 시, 코드와 함께 반드시 **출력 결과도 첨부하기**
- 완벽하지 않더라도 업로드 바랍니다~😊


<br>

## 실습 설명
### ◼ 데이터

- [data-02-stock_daily.csv](https://github.com/deeplearningzerotoall/PyTorch/blob/master/data-02-stock_daily.csv) | [**다운로드**](https://github.com/Artinto/Python_and_AI_Study/files/6494028/stock.zip)

- **정보**
   | Open | High | Low | Volume | Close |
   |:--:|:--:|:--:|:--:|:--:|
   |시가 | 최고가 | 최저가 | 거래량 | 종가 |
   
   - 위 데이터는 하루 간격의 주식시장 데이터입니다.


   - 맨 윗줄은 가장 최근의 데이터를 의미합니다. <br> (학습 시 고려하여 데이터를 구성하시기 바랍니다.)

<br>

### ◼ 실습 목표
- **지난 N(N≤14)일의 주식 데이터를 보고 다음날의 종가를 예측하는 모델을 작성**

- RNN/LSTM/GRU(선택) 모델 구축
- 위 데이터의 가장 오랜된 날부터 70%를 train으로, 나머지를 test 데이터로 사용
- train 데이터로 모델 학습, test 데이터로 모델 성능 평가
- 모델을 학습하여 1 epoch 마다 test 데이터에 대한 loss 출력
