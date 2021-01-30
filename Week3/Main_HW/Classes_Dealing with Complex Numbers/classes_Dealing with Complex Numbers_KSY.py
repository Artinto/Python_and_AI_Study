import math

#real 실수 imaginary 허수 
# Python 3

class Complex(object):

    def __init__(self, real, imaginary):
        self.real = real
        self.imaginary = imaginary
        #print("real : %0.2f imaginary: %0.2f" % (self.real,self.imaginary))
        
    def __add__(self, no):
        real = self.real + no.real
        imaginary = self.imaginary + no.imaginary
        return Complex(real,imaginary)
        # C = Complex(self, no)
        # C.real = self.real + no.real
        # C.imaginary = self.imaginary + no.imaginary
        # return C
        # # return print("%.2f+%.2fi"%(self.real, self.imaginary))
        
    def __sub__(self, no):
        real = self.real - no.real
        imaginary = self.imaginary - no.imaginary
        return Complex(real,imaginary)
        
    def __mul__(self, no):
        real = self.real * no.real - self.imaginary * no.imaginary
        imaginary = self.real * no.imaginary + self.imaginary * no.real
        return Complex(real,imaginary)

    def __truediv__(self, no):
        x = no.real ** 2 + no.imaginary ** 2
        a=(self.real*no.real+self.imaginary*no.imaginary)/x
        b=(-no.imaginary*self.real+self.imaginary*no.real)/x
        return Complex(a,b)
        

    def mod(self):
        real= pow(self.real**2+self.imaginary**2,0.5)
        return Complex(real, 0)

    def __str__(self):
        if self.imaginary == 0:
            result = "%.2f+0.00i" % (self.real)
        elif self.real == 0:
            if self.imaginary >= 0:
                result = "0.00+%.2fi" % (self.imaginary)
            else:
                result = "0.00-%.2fi" % (abs(self.imaginary))
        elif self.imaginary > 0:
            result = "%.2f+%.2fi" % (self.real, self.imaginary)
        else:
            result = "%.2f-%.2fi" % (self.real, abs(self.imaginary))
        return result

if __name__ == '__main__':
    c = map(float, input().split())
    d = map(float, input().split())
    x = Complex(*c)
    y = Complex(*d)
    print(*map(str, [x+y, x-y, x*y, x/y, x.mod(), y.mod()]), sep='\n')
