import matplotlib.pyplot as plt
import numpy as np

def FirstOctan(r):
    ret = []
    x = 0
    y = r
    p = 1 - r
    
    while x <= y:
        ret.append((x,y))
        x += 1
        if p < 0:
            p = p + 2 * x + 1
        else:
            y -= 1
            p = p + 2 * x - 2 * y + 1
    return ret

def DrawCircle(xc,yc,r):
    Octane = FirstOctan(r)
    for point in Octane:
        x = point[0] 
        y = point[1] 
        plt.scatter([ x+xc , y+yc , x+xc , -y+yc , -x+xc , y+yc , -x+xc , -y+yc ] , [ y+yc , x+xc , -y+yc , x+xc , y+yc , -x+xc , -y+yc , -x+xc ])

    plt.axis('equal')
    print(Octane)
    plt.show()
    
    
if __name__ == "__main__":
    DrawCircle(2,3,30)