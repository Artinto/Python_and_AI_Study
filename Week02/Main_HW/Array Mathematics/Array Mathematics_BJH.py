if __name__ == '__main__':
    
    m,n = map(int,input().split())
    list1 = []
    list2 = []
    total = []
    symbol = ['+', '-','*','/','%','**']
    k = 0
    num = input("첫번째 리스트의 값들을 입력하시오").split()
    while len(num) != n:
        num= input("다시 입력하시오").split()
    for i in num:
        list1.append(i)
        
    num = input("두번째 리스트의 값들을 입력하시오").split()
    while len(num) != n:
        num= input("다시 입력하시오").split()
    for i in num:
        list2.append(i)
    
    for k in symbol:
         for i in range(0,n):
            if k =='+':
                total.append(int(list1[i]) + int(list2[i]))
            elif k =='-':
                total.append(int(list1[i]) - int(list2[i]))
            elif k =='*':
                total.append(int(list1[i]) * int(list2[i]))
            elif k =='/':
                total.append(int(int(list1[i]) / int(list2[i])))
            elif k =='%':
                total.append(int(list1[i]) % int(list2[i]))
            elif k =='**':
                total.append(int(list1[i]) ** int(list2[i]))
         print(total)
         del total[0:]
