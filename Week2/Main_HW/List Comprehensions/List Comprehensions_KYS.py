x = int(input())
y = int(input())
z = int(input())
n = int(input())

res = []
arr = []
idx = [x, y, z]
def dfs(cnt):
    if cnt == 3:
        if(sum(arr) != n):
            return [arr[0], arr[1], arr[2]]
            #res.append(arr)
        return
        
    for i in range (idx[cnt] + 1):
        arr.append(i)
        res.append(dfs(cnt + 1))
        arr.pop()

dfs(0)
print(list(filter(None, res)))