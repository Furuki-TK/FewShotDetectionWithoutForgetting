f = open("id2label.txt",'r')

buf = f.read().split()

aa = {}
for n in buf:
    i,d = n.split(',')
    aa[int(i)] = d

print(aa[0])

f.close()
