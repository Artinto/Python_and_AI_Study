if __name__ == '__main__':
    
    students={}
    n = input()
    for i in range(0,int(n)):
        line = input().split()
        name = line[0]
        scores = line[1:]
        students[name] = scores
        
    print("""원하는 학생의 이름을 말하시오""")
    query_name = input()
    sum =0
    
    for i in students[query_name]:
        sum += int(i)
    print(sum/len(students[query_name]))
    




