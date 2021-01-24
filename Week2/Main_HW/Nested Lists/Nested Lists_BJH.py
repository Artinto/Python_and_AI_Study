records=[]
scores=[]
#입력 부분
for _ in range(int(input())):
    student = input()
    score = float(input())
    records.append([student,score])
    scores.append(score)
# 점수대를 정렬
scores.sort()
# 그 중 2번째 큰 값 가져오기
low= scores[1]
#2번째 큰 값과 같은 점수의 사람 출력하
for i,j in records:
    if low== j:
        print(i)




