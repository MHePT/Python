def CSM(file1,file2):
    
    wordlist = vector1 = vector2 = []
    file1 = open(file1,'rt')
    file2 = open(file2,'rt')

    for line in file1:
        words = line.split()
        for word in words:
            if word not in wordlist:
                wordlist.append(word)
         
    for line in file2:
        words = line.split()
        for word in words:
            if word not in wordlist:
                wordlist.append(word)
                
    for word in worldlist:
        
        counter1 = counter2 = 0
        
        for wr in file1:
            if word == wr:
                counter1 += 1
                
        for wr in file2:
            if word == wr:
                counter2 += 1
                
        vector1.append(counter1)   
        vector2.append(counter1) 
        
    dot = index = magVector1 = magVector2 = 0
    for i in vector1:
        magVector1 += i**2
        magVector2 += vector2[index] ** 2
        dot += i * vector2[index]
        index += 1
        
    magVector2 = magVector2**0.5
    magVector1 = magVector1**0.5
    
    return dot/(magVector2*magVector1)


if __name__ == "__main__":
    CSM("C:\Users\moham\Desktop\test1.txt","C:\Users\moham\Desktop\test2.txt")
             
    