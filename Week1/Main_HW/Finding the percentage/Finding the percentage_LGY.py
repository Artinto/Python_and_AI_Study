# name:[mark] 형태의 딕서너리를 만들고, 해당 학생의 평균 점수를 print하라.(소수점두째자리까지)
'''
  line_1 : 학생수(n)
  n명의 name, 3개의 marks

  final line : query_name(해당학생의 평균점수를 print)
''' 

if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split() 
        # *line : 첫번째 stub은 name, 나머지 input은 line에 담는다는 의미
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input() #딕셔너리 만드는 것까지 모두 되어있음.
    
    
    #만들어진 dic에서 특정 name을 통해 marks 불러와 평균계산하기
    average_score = sum(student_marks[query_name])/3 # 3은 고정된 수
    print('%.2f' % average_score) #둘째자리까지
    
    
 
