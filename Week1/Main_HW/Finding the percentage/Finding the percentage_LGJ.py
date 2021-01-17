if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    student_average = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
        add = sum(scores)
        average =  (add)/3
        student_average [name] = average
    query_name = input()
    print('%.2f' %student_average[query_name])
