
if __name__ == '__main__':
    records = []
    for _ in range(int(input())):
        name = input()
        score = float(input())
        records.append([name,score])
    
    Scores = []
    for i,e in records:
        Scores.append(e) 

    MinValue = min(Scores) 

    # 가장 작은 값 삭제
    while(MinValue in Scores):            # 가장 작은 값을 가진 원소 Scores에서 삭제
        Scores.remove(MinValue)
        
    for i in range(len(records)):
        if(i<len(records)):
            if(records[i][1] == MinValue): 
                records.remove(records[i]) # 가장 작은 값을 가진 원소 Records에서 삭제
            
    # 두 번째로 작은 값 
    SecondMin = min(Scores)
    
    LappedScore = []
    for i in range(len(records)):
        if(i<len(records)):
            if(records[i][1] == SecondMin):
                LappedScore.append(records[i][0])  # 이름 

    LappedScore.sort()
    
    for i,e in enumerate(LappedScore):
        print(LappedScore[i])
    
