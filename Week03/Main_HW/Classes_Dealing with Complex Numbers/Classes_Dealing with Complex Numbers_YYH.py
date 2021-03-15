import math

class Complex(object):
    def __init__(self, real, imag):
        self.real=real
        self.imag=imag
        
    def __add__(self, other):
        obj=Complex(self.real + other.real, self.imag + other.imag)
        return obj
    def __sub__(self, other):
        obj=Complex(self.real - other.real, self.imag - other.imag)
        return obj
    def __mul__(self, other):
        obj=Complex(self.real * other.real-self.imag * other.imag,self.imag * other.real+self.real*other.imag)
        return obj
    def __truediv__(self, other):
        obj=Complex((self.real*other.real+self.imag*other.imag)/(other.real**2+other.imag**2),(self.imag*other.real-self.real*other.imag)/(other.real**2+other.imag**2))
        return obj
    def mod(self):
        obj=Complex(math.sqrt(self.real**2+self.imag**2),0)
        return obj
    def __str__(self):
        if self.imag == 0:
            result = "%.2f+0.00i" % (self.real)
        elif self.real == 0:
            if self.imag >= 0:
                result = "0.00+%.2fi" % (self.imag)
            else:
                result = "0.00-%.2fi" % (abs(self.imag))
        elif self.imag > 0:
            result = "%.2f+%.2fi" % (self.real, self.imag)
        else:
            result = "%.2f-%.2fi" % (self.real, abs(self.imag))
        return result

if __name__ == '__main__':
    c = map(float, input().split())
    d = map(float, input().split())
    x = Complex(*c)
    y = Complex(*d)
    print(*map(str, [x+y, x-y, x*y, x/y, x.mod(), y.mod()]), sep='\n')
