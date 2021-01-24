if __name__ == '__main__':
    
    student=[]
    sscore=[]
    
    for _ in range(int(input())):
        name = input()
        score = float(input())
       
    
        student.append([name,score])
        sscore.append(score)
        
        sscore.sort()
    new_score=[]
    for i in sscore:
        
     if i not in new_score:
        new_score.append(i)
        
    second=new_score[1]
    
    for i,j in student:
        if j== second:
            print(i)
  
