Arr = []
Score = []
Final = []
Name = []

for _ in range(int(input())):
    name = input()
    score = float(input())
    arr = [name,score]
    Arr.append(arr)#리스트Arr에 리스트arr[name,score]를 넣음
    Score.append(score)#리스트Score에 score를 넣음

Score.sort()#점수를 오름차순으로 분류

for i in Score:#Score리스트에서 겹치는 score를 제거
    if i not in Final:
        Final.append(i)

seclow = Final[1]#두번째로 작은 점수 seclow에 대입

for j in range(0,len(Arr)):#Arr에서 seclow와 일치하는 리스트를 찾아 그 리스트의 name값을 Name리스트에 대입
    if seclow in Arr[j]:
        t=Arr[j][0]
        Name.append(t)

Name.sort()#사전순으로 이름 나열
for k in range(0,len(Name)):#이름 출력
    print(Name[k])
