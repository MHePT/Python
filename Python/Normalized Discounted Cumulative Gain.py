import math

def nDCG(k,g):
    table = [[0 for _ in range(8)] for _ in range(len(k))]
    
    ig = g.copy()
    ig.sort(reverse = True)
    total = 0
    totali = 0
    for i in range(len(k)):
        table[i][0] = k[i]
        table[i][1] = g[i]
        table[i][4] = ig[i]
        
        if k[i] > 1:
            table[i][2] = g[i]/math.log2(k[i])
            total += g[i]/math.log2(k[i])
            
            table[i][5] = ig[i]/math.log2(k[i])
            totali += ig[i]/math.log2(k[i])
        else:
            table[i][2] = g[i]
            total += g[i]
            
            table[i][5] = ig[i]
            totali += ig[i]
            
        
    table[9][3] = total
    table[9][6] = totali   
    table[9][7] = total/ totali
    
    return table
       
        

if __name__ == "__main__":
    x = nDCG( range(1,11) , [3,2,3,0,0,1,2,2,3,0] )
    for row in x:
        for el in row:
            if len(str(el)) < 17:
                print(el , end = " "*(18-len(str(el))) +",")
            else:
                print(el, end = ",")
        print()