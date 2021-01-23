import numpy

num = input()

num = str(num)
num = num.split()

list1 = list(map(int,num))

my_array = numpy.array(list1)
print(numpy.reshape(my_array,(3,3)))
