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
            res.append(arr.copy())
        return
        
    for i in range (idx[cnt] + 1):
        arr.append(i)
        dfs(cnt + 1)
        arr.pop()

dfs(0)
print(res)