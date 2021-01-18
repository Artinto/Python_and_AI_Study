if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()           # Krishna 67 68 69
        scores = list(map(float, line))         # [67.0, 68.0, 69.0]
        student_marks[name] = scores
    
    # student_marks = {'Krishna': [67.0, 68.0, 69.0], 'Arjun': [70.0, 98.0, 63.0], 'Malika': [52.0, 56.0, 60.0]}
    
    query_name = input()
    
    q_scores = student_marks[query_name]
    print( '{:.2f}'.format( sum(q_scores)/len(q_scores) ) )