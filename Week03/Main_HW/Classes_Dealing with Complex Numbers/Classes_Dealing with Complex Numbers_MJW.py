import math

class Complex(object):
    def __init__(self, real, imaginary):
        self.real = real
        self.imaginary = imaginary
        
    def __add__(self, no):
        return Complex( self.real+no.real , self.imaginary+no.imaginary )
        
    def __sub__(self, no):
        return Complex( self.real-no.real , self.imaginary-no.imaginary )
        
    def __mul__(self, no):
        r = (self.real*no.real) - (self.imaginary*no.imaginary)
        i = (self.real*no.imaginary) + (self.imaginary*no.real)
        return Complex(r, i)

    def __truediv__(self, no):
        a = Complex(self.real, self.imaginary) * Complex(no.real, -no.imaginary) # 분자
        b = no.real ** 2 + no.imaginary ** 2                                     # 분모
        r = a.real / b
        i = a.imaginary / b
        return Complex(r, i)

    def mod(self):
        r = (self.real**2 + self.imaginary**2) ** 0.5
        return Complex(r, 0)
        
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
