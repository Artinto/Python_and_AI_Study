#key, value 나눠 sort. dict에서 찾아서 프린트

if __name__ == '__main__':
    student = {}
    ans = []

    for _ in range(int(input())):
        name = input()
        score = float(input()) # input값을 받아서
        student[name] = score  # dictionary로 저장
         
    names = student.keys()     # key, value로 각각 리스트 생성
    scores = student.values()
    
    S = sorted(list(set(scores))) # set으로 중복제거
    second = S[1]  # 두번째로 낮은 점수
    
    A = [name_ for name_, score_ in student.items() if score_ == second]    
    # student라는 딕셔너리에서 value(score)값이 '두번째로 낮은 점수(second)'와 같다면(if),
    # 그때 key(name)값을 A라는 list로 만듦.
    A.sort() # 알파벳 순으로 정렬하기 위해
            
    for i in A: # 하나씩 프린트
        print(i)
		
		
------------------------------------------------------------------------------
# dict 정렬 사용, 찾아서 프린트

if __name__ == '__main__':
    student = {}
    ans = []

    for _ in range(int(input())):
        name = input()
        score = float(input()) # input값을 받아서
        student[name] = score  # dictionary로 저장
         
    scores = student.values()
    
    S = sorted(list(set(scores))) # set으로 중복제거
    second = S[1]  # 두번째로 낮은 점수
    
    
    Student = dict(sorted(student.items(), key = lambda x : x[0]))
    # lambda x : x[0]
    # x라는 인자가 들어오면 x[0]을 출력한다. >> student.keys()
    
    # sort을 거치면 tuple형태로 출력되서 dict형태로 바꿈 
    # Student : student를 알파벳순으로 정렬한 dict
    
    
    for name_, score_ in Student.items(): #알파벳순으로 정렬된거 중에
        if score_ == second: # 찾는 점수와 같으면
            
            print(name_) # 그때 key값 출력
   






