#3가지경우에 해당하는 윤년계산하기
#1. 윤년은 4로 나눠진다
#2. 100으로 나눠지는 해는 윤년이 아니다.
#3. 400으로 나눠지는 해는 윤년이다. 

def is_leap(year):
    leap = False
    
    if year%400 == 0:
        leap = True
    elif year%100 == 0:
        leap = False
    elif year%4 == 0:
        leap = True
        
    return leap
   
    
year = int(input())
print(is_leap(year))
