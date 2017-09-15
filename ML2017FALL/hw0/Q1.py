import sys


wordfile = sys.argv[1]
f = open(wordfile)
words = f.read().split()
f.close()

l = []
l.append([])
l[0].append(words[0])
l[0].append(0)
l[0].append(1)

for i in range(1,len(words)):
    for j in range(len(l)):
        if words[i] in l[j]:
            l[j][2] +=1
            break
    if j+1 == len(l):   
        l.append([])
        l[j+1].append(words[i])
        l[j+1].append(j+1)
        l[j+1].append(1)
#f = open('Q1.txt','w')
for i in l:
    k=' '.join([str(j) for j in i])
    print (k)
    #f.write(k+"\n")
#f.close()   

