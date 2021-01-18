def is_leap(year):
    leap = False
    
    if not year % 400:      # (=) year % 400 == 0
        leap = True
    elif not year % 100:
        leap = False
    elif not year % 4:
        leap = True
    
    return leap

year = int(input())
print(is_leap(year))