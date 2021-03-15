import math

class Complex(object):
    def __init__(self, real, imaginary):
        self.real=real # 실수부
        self.imaginary=imaginary # 허수부
        
    def __add__(self, no):
        obj= Complex(self.real+no.real,self.imaginary+no.imaginary)
        return obj
    
    def __sub__(self, no):
        obj= Complex(self.real-no.real,self.imaginary-no.imaginary)
        return obj
    
    def __mul__(self, no):
        obj= Complex(self.real*no.real-self.imaginary*no.imaginary, self.real*no.imaginary+self.imaginary*no.real)
        return obj

    def __truediv__(self, no):
        obj= Complex((self.real*no.real-self.imaginary*(-no.imaginary))/(no.real*no.real+no.imaginary*no.imaginary),(self.imaginary*no.real+self.real*(-no.imaginary))/(no.real*no.real+no.imaginary*no.imaginary))
        return obj
        
    def mod(self):
        obj= Complex((self.real**2+self.imaginary**2)**0.5,0)
        return obj

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
