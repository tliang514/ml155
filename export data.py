f=open('demo.txt','w')
for i in range(1,1356):
    #k=str(list[i-1])
    k=str(i)+","+str(y_test[i-1])
    f.write(k+"\n")
f.close()
