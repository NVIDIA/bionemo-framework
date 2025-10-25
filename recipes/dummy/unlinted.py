def CalculateStuff(data,x=[], flag=True):
    import os
    list=[]
    dict={}
    for i in range(0,len(data)):
        if flag==True:
            if data[i]!=None:
                if type(data[i])==int:
                    try:
                        result=data[i]*2+5-3/2
                        list.append(result)
                        y=10
                        z=20
                        a=30
                    except:
                        pass
                else:
                  dict[i]=data[i]
        elif flag==False:
                x.append(data[i])
    return list,dict
