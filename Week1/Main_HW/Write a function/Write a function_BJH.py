if __name__ == '__main__':
    
    year = int(input())
    
    if year % 4 ==0 and year % 100 !=0:
        print("""leap year""")
    elif year % 100 ==0 and year % 400 !=0:
        print("""false""")
    elif year % 400 ==0:
        print("""leap year""")
    else:
        print("""false""")
       
    
    




