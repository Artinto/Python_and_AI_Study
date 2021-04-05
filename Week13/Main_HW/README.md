# Custom 데이터로 CNN 모델 학습 실습


### ※ 준수사항
- .py 혹은 .ipynb 파일로 업로드 시, 코드와 함께 반드시 **출력 결과도 첨부하기**
- 완벽하지 않더라도 업로드 바랍니다~😊


<br>

## 실습 설명
### ◼ 데이터

- [**데이터셋**](https://github.com/Artinto/Python_and_AI_Study/tree/main/Week13/custom_dataset)

- **정보**
   - **4개의 클래스 - label 정보 : 0(오징어짬뽕), 1(비빔면), 2(짜왕), 3(무파마)**
   
      |    | 0  |  1 | 2 | 3 |
      |:--:|:--:|:--:|:--:|:--:|
      |train set | 487 | 511 | 475 | 476 |
      |test set | 25 | 25 | 25 | 25 |


<br>

### ◼ 실습 목표
- custom 데이터 로드하기
- CNN 모델 구축 
- train 데이터로 모델 학습, test 데이터로 모델 성능 평가
- 모델을 학습하여 1 epoch 마다 loss와 accuracy 출력
