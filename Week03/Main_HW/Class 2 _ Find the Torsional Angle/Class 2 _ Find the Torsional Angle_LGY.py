import math

class Points(object):
    def __init__(self, x, y, z): #들어오면 우선 설정
        self.x = x
        self.y = y
        self.z = z
        

    def __sub__(self, B): # '-'하면 작동 / A(self) - B
        ans_x = self.x - B.x
        ans_y = self.y - B.y
        ans_z = self.z - B.z
        return Points(ans_x, ans_y,ans_z)
    
    def dot(self, B): # 내적 /
        ans_x = self.x * B.x
        ans_y = self.y * B.y
        ans_z = self.z * B.z
        answer = ans_x + ans_y + ans_z
        return answer

    def cross(self, B): # 외적 / A.cross(B) (b-a : A / c-d : B)
        ans_x = self.x*B.y - B.x*self.y
        ans_y = self.y*B.z - B.y*self.z
        ans_z = self.z*B.x - B.z*self.x
        #(외적계산) 사루스법칙
        return Points(ans_x, ans_y,ans_z)
        
        
    def absolute(self): # 크기계산
        return pow((self.x ** 2 + self.y ** 2 + self.z ** 2), 0.5)

        
# Torsional Angle : 이면각
# 3-dimensional Cartesian coordinate system : 3차원 좌표계
if __name__ == '__main__':
    points = list()
    for i in range(4):
        a = list(map(float, input().split()))
        points.append(a)

    a, b, c, d = Points(*points[0]), Points(*points[1]), Points(*points[2]), Points(*points[3])
    x = (b - a).cross(c - b)
    y = (c - b).cross(d - c)
    angle = math.acos(x.dot(y) / (x.absolute() * y.absolute()))

    print("%.2f" % math.degrees(angle))
