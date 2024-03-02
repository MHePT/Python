def draw_line(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    m = dy/dx
    fx = x1 
    fy = y1
    mylist = [[0 for i in range(max(y1,y2)+1)] for j in range(max(x1,x2)+1)]
    mark = 1
    
    mylist[round(fx)][round(fy)] = mark

    if m < 1:
        while fx != x2:
            fx += 1
            fy += m
            mylist[round(fx)][round(fy)] = mark
    elif m > 1:
        while fy != y2:
            fy += 1
            fx += 1/m
            mylist[round(fx)][round(fy)] = mark
    else:
        while fy != y2 and fx != x2:
            fx += 1
            fy += 1
            mylist[round(fx)][round(fy)] = mark
    

    for row in mylist:
        print(row)

# Example usage
draw_line(10, 20, 25, 30)
