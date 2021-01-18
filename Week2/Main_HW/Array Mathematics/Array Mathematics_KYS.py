import numpy

n, m = map(int, input().split())

arr_a = [list(map(int, input().split())) for _ in range(n)]
arr_b = [list(map(int, input().split())) for _ in range(n)]

num_a = numpy.array(arr_a)
num_b = numpy.array(arr_b)

print(numpy.add(num_a, num_b))
print(numpy.subtract(num_a, num_b))
print(numpy.multiply(num_a, num_b))
print(numpy.floor_divide(num_a, num_b)) #나눗셈 내림 처리
print(numpy.mod(num_a, num_b))
print(numpy.power(num_a, num_b))