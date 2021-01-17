# runner-up score : 2위 점수
# 모든 점수를 list에 저장하고, runner-up score를 찾아라

if __name__ == '__main__':
    n = int(input())
    arr = list(set(map(int, input().split())))
    #set으로 중복제거 > list로 변환 (sort사용하기 위해서)
    
    arr.sort() #정렬[작은거 >> 큰거]
    print(arr[-2]) # 뒤에서 2번째(runner-up score) print
 
  
