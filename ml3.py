import pandas as pd
import numpy as np 

data=pd.DataFrame(data=pd.read_csv('ds1.csv'))
concepts=np.array(data.iloc[:,:-1])
target=np.array(data.iloc[:,-1])


def learn(concepts,target):
    specific_h=concepts[0].copy()
    general_h=[["?" for i in range(len(specific_h))] for i in range(len(specific_h))]

    for i,h in enumerate(concepts):
        if target[i]=='yes':
            for x in range(len(specific_h)):
                if h[x]!=specific_h[x]:
                    specific_h[x]='?'
                    general_h[x][x]='?'

        else:
            for x in range(len(specific_h)):
                if h[x]!=specific_h[x]:
                    general_h[x][x]=specific_h[x]
                else:
                    general_h[x][x]='?'
            
        print("iterations:",i+1)
        print("specific_h:  ")
        print(specific_h)
        print("general_h: ",)
        print(general_h)
        print("\n")
    
    indecis=[i for i,val in enumerate(general_h) if val==['?','?','?','?','?','?']]
    print(indecis)
     
    for i in indecis:
        general_h.remove(['?','?','?','?','?','?'])
    
    return specific_h,general_h

final_s,final_g=learn(concepts,target)
print("fanal S: ",final_s)
print("fanal g: ",final_g)


            