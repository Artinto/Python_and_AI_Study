def is_leap(year):
    if(year%4 or year%100==0) and year%400:return False
    return True

year = int(input())
print(is_leap(year))