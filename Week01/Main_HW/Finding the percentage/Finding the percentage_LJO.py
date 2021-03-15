if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    sum=0
    if query_name in student_marks:
        b=len(student_marks[query_name])
        for i in student_marks[query_name]:
            sum+=i
    avg=float(sum/b)
    print("%.2f" %avg)
