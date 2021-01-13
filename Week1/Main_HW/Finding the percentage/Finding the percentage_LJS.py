if __name__ == '__main__':
    n = int(input())
    student_marks = {} 
    
    # { Name:[Score ~], ~} 
    for _ in range(n): 
        name, *line = input().split()   # list Unpacking
        scores = list(map(float, line)) # To float -> Packing
        student_marks[name] = scores  
    query_name = input()
    
    Sum = 0
    
    for i in student_marks[query_name]:
        Sum += i 
    
    print('%.2f' %(Sum/3))
