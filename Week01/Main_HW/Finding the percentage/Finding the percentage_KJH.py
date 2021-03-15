if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    
     # Finding the percentage in Python - Hacker Rank Solution START
    output = list(student_marks[query_name])
    per = sum(output)/len(output);
    print("%.2f" % per);
    # Finding the percentage in Python - Hacker Rank Solution END
