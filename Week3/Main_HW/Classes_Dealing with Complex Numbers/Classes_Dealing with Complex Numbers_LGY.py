import math

class Complex(object):   # 복소수
    
    def __init__(self, real, imaginary):
        self.real = real  #실수
        self.imaginary = imaginary   #허수

    def __add__(self, B): # A+B
        a = self.real + B.real
        b = self.imaginary + B.imaginary
        return Complex(a, b)
        
    def __sub__(self, B): # A-B
        a = self.real - B.real
        b = self.imaginary - B.imaginary
        return Complex(a, b)
        
    def __mul__(self, B): # A*B
        a = self.real*B.real - self.imaginary*B.imaginary
        b= self.real*B.imaginary + self.imaginary*B.real
        return Complex(a, b)

    def __truediv__(self, no):
        a = self.real*no.real + self.imaginary*no.imaginary
        b= - self.real*no.imaginary + self.imaginary*no.real
        c = no.real**2 + no.imaginary**2
        return Complex(a/c, b/c)

    def mod(self):
        a = self.real**2 + self.imaginary**2
        b = math.sqrt(a)
        return Complex(b,0)

    def __str__(self): # __str__ 인스턴스자체를 출력할 때 형식을 지정하는 함수

        # n+0.00i에서
        if self.imaginary == 0: #허수가 0이라면(실수)
            result = "%.2f+0.00i" % (self.real) # n + 0.00i라는 수가 된다.(복소수형으로)
            #실수의 복소수화
        
        # 0 + ni에서 
        elif self.real == 0: #실수가 0일때

            #0 + ni
            if self.imaginary >= 0: # 허수가 (+)값으로 존재할 경우
                result = "0.00+%.2fi" % (self.imaginary)
            
            #0 - ni
            else: # 허수가 (-)값으로 존재할 경우
                result = "0.00-%.2fi" % (abs(self.imaginary)) #imaginary의 절대값해서 양수값으로 대입
                

        # m + ni
        elif self.imaginary > 0: #실수가 양수값일때
            result = "%.2f+%.2fi" % (self.real, self.imaginary)

        # m - ni
        else:
            result = "%.2f-%.2fi" % (self.real, abs(self.imaginary))
        return result
        

if __name__ == '__main__':
    c = map(float, input().split())
    d = map(float, input().split())
    x = Complex(*c)
    y = Complex(*d)
    print(*map(str, [x+y, x-y, x*y, x/y, x.mod(), y.mod()]), sep='\n')
