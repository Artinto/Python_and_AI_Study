if __name__ == '__main__':
    namebox=[]
    scorebox=[]
    for _ in range(int(input())):
        name = input()
        score = float(input())
        namebox.append([name,score])
        scorebox.append(score)
    scorebox.sort()
    ed_score = scorebox[1]
    namebox.sort()
    for name, score in namebox:
        if score == ed_score:
            print(name)
