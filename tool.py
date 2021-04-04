import json

"""
功能：发现二者之间的差异
01.set 中二者的差是不同的。即a-b != b-a
"""
def getanc(path_1,path_2):
    with open(path_1,'r') as f:
        cont = json.load(f)
        key1 = cont.keys()
        key1 = set(key1)
    with open(path_2,'r') as f:
        cont = json.load(f)
        key2 = cont.keys()
        key2 = set(key2)
    print(key1)
    print(key2)
    print("-----")
    print("交集：",key1 & key2)
    print("------")
    print("差集：",key1 - key2)
    print("------")
    print("差集",key2-key1)



path_1 = './data/predicate2id.json'
path_2 = './data/predicate2id_2.json'
getanc(path_1,path_2)
