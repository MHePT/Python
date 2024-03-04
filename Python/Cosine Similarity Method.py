def CSM(file1,file2):
    
    wordlist = []
    vector1 = []
    vector2 = []
    lines1 = []
    lines2 = []
    file1 = open(file1,'rt')
    file2 = open(file2,'rt')

    for line in file1:
        words = line.split()
        lines1.append(line)
        for word in words:
            if word.lower() not in wordlist:
                wordlist.append(word.lower())

    for line in file2:
        words = line.split()
        lines2.append(line)
        for word in words:
            if word.lower() not in wordlist:
                wordlist.append(word.lower())

    print(wordlist)
    file1.close()
    file2.close()
    
    for word in wordlist:
        if word == 0:
            break
        
        counter1 = 0
        counter2 = 0
       
        for line in lines1:
            l = line.split()
            for wr in l:
                if word.lower() == wr.lower():  
                    counter1 += 1
   
        for line in lines2:
            l = line.split()
            for wr in l:
                if word.lower() == wr.lower():
                    counter2 += 1
        
        vector1.append(counter1)   
        vector2.append(counter2) 

    print(vector1)
    print(vector2)
 
    
    dot = 0
    index = 0
    magVector1 = 0
    magVector2 = 0
    for i in vector1:
        magVector1 += i**2
        magVector2 += vector2[index] ** 2
        dot += i * vector2[index]
        index += 1
        
    magVector2 = magVector2**0.5
    magVector1 = magVector1**0.5
    file1.close()
    file2.close()

    return dot/(magVector2*magVector1)
             
    