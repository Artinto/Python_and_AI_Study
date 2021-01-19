arr = {}
val = float('inf')
for _ in range(int(input())):
    name = input()
    score = float(input())
    arr[name] = score
    val = min(val, score)

tmp = sorted(arr) # get sorted key "return tuple not dictionary"
arr2 = {}
val2 = float('inf')

for name in tmp:
    if(arr[name] != val):
        arr2[name] = arr[name]
        val2 = min(val2, arr[name])

for name, score in arr2.items():
    if(score == val2):
        print(name)