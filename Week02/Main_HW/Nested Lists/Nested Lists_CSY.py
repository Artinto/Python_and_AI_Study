if __name__ == '__main__':
    
    l=[]
    scores=[]
    for _ in range(int(input())):
        name = input()
        score = float(input())
        l.append([name,score])
        scores.append(score)
    l.sort()
    scores=list(set(scores))
    scores.sort()
    second_min=scores[1]
    for i in range(len(l)):
        if second_min==l[i][1]:
            print(l[i][0])
