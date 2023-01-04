import math 
import pandas as pd
from collections import Counter
from pprint import pprint

df_teinnis=pd.DataFrame(data=pd.read_csv('ds2.csv'))
print(df_teinnis)

def entropy(a_list):
    cnt=Counter(x for x in a_list)
    print("no and yes is: ",a_list.name,cnt)
    no_of_instances=len(a_list)*1.0
    probs=[x/no_of_instances for x in cnt.values()]
    return sum([-probs*math.log(probs,2) for probs in probs])

print(df_teinnis['playtennis'])
total_entropy=entropy(df_teinnis['playtennis'])
print("total entropy is :",total_entropy)

def information_gain(df,split_attribute_name,target_attribute_name,trace=0):
    print("information gain for:",split_attribute_name)
    df_split=df.groupby(split_attribute_name)
    for name,group in df_split:
        print(name)
        print(group)
    nob=len(df.index)
    df_agg1=df_split.agg({target_attribute_name:lambda x:entropy(x)})
    df_agg2=df_split.agg({target_attribute_name:lambda x:len(x)/nob})
    df_agg1.columns=['entropy']
    df_agg2.columns=['proportion']
    new_entropy=sum(df_agg1['entropy']*df_agg2['proportion'])
    old_entropy=entropy(df[target_attribute_name])
    return old_entropy-new_entropy

print("information gain for outlook:",information_gain(df_teinnis,'outlook','playtennis'))
print("information gain for temperature:",information_gain(df_teinnis,'temperature','playtennis'))
print("information gain for humidity:",information_gain(df_teinnis,'humidity','playtennis'))

def id3(df,target_attribute_name,atribute_names,default_class=None):
    cnt=Counter(x for x in df[target_attribute_name])
    if len(cnt)==1:
        return next(iter(cnt))
    elif df.empty or (not atribute_names):
        return default_class
    else:
        default_class=max(cnt.keys())
        gainz=[information_gain(df,attr,target_attribute_name) for attr in atribute_names]
        index_of_max=gainz.index(max(gainz))
        best_attr=atribute_names[index_of_max]
        tree={best_attr:{ }}
        remaining_attribute_name=[i for i in atribute_names if i!=best_attr]
    for attr_val,data_subset in df.groupby(best_attr):
        subtree=id3(data_subset,target_attribute_name,remaining_attribute_name,default_class)
        tree[best_attr][attr_val]=subtree
    return tree



print("\n")
atribute_names=list(df_teinnis.columns)
print("list of attribute names",atribute_names)
atribute_names.remove('playtennis')
print("predicting attribute names",atribute_names)
tree=id3(df_teinnis,'playtennis',atribute_names)
pprint("decision tree")
pprint(tree)



def clasify(instance,tree,default=None):
    attribute=next(iter(tree))
    if instance[attribute] in tree[attribute].keys():
        result=tree[attribute][instance[attribute]]
        if isinstance(result,dict):
            return clasify(instance,result)
        else:
            return result
    else:
        return default

df_new=pd.read_csv('playtennis.csv')
df_new['pridicted']=df_new.apply(clasify,axis=1,args=(tree,'?'))
print(df_new)