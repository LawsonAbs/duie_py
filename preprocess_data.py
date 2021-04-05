import json

"""
预处理数据，得到 id2subject.json
"""
def get_id2subject(in_path,out_path):
    subject_map={}        
    with open(in_path,'r') as f:
        cont = json.load(f)
        index = 2
        for row in cont:
            if row['subject_type'] not in subject_map.values():
                subject_map[index] = row['subject_type']
                index+=1
    
    with open(out_path,'w',encoding="utf-8") as f:
        json.dump(subject_map,f,ensure_ascii=False)


"""
预处理数据，得到 subject2id.json, object2id.json 文件
"""
def get_subject2id(in_path,out_path):
    subject_map={}        
    with open(in_path,'r') as f:
        cont = json.load(f)
        index = 2
        for row in cont:
            if row['subject_type'] not in subject_map.keys():
                subject_map[row['subject_type']] = index
                index+=1
    
    with open(out_path,'w',encoding="utf-8") as f:
        json.dump(subject_map,f,ensure_ascii=False)



"""
预处理数据，得到 object2id.json 文件
"""
def get_object2id(in_path,out_path):
    object_map={}        
    with open(in_path,'r') as f:
        cont = json.load(f)
        index = 2
        for row in cont:
            vals = list(row['object_type'].values())            
            for val in vals:
                if val not in object_map.keys():
                    object_map[val] = index
                    index+=1
    with open(out_path,'w',encoding="utf-8") as f:
        json.dump(object_map,f,ensure_ascii=False)


"""
预处理数据，得到 id2object.json 文件
"""
def get_id2object(in_path,out_path):
    object_map={}        
    with open(in_path,'r') as f:
        cont = json.load(f)
        index = 2
        for row in cont:
            vals = list(row['object_type'].values())            
            for val in vals:
                if val not in object_map.values():
                    object_map[index] = val
                    index+=1
    with open(out_path,'w',encoding="utf-8") as f:
        json.dump(object_map,f,ensure_ascii=False)



'''
得到relation2id.json
'''
def get_relation(in_path,out_path):
    relation_map={}
    with open(in_path,'r') as f:
        cont = json.load(f)
        index = 0
        for row in cont:
            relation = row['predicate']
            relation_map[relation] = index
            index+=1
    with open(out_path,'w',encoding="utf-8") as f:
        json.dump(relation_map,f,ensure_ascii=False)


'''
得到 id2relation.json
'''
def get_id2relation(in_path,out_path):
    relation_map={}
    with open(in_path,'r') as f:
        cont = json.load(f)
        index = 2
        for row in cont:
            relation = row['predicate']
            relation_map[index] = relation
            index+=1
    with open(out_path,'w',encoding="utf-8") as f:
        json.dump(relation_map,f,ensure_ascii=False)



if __name__ == "__main__":
    in_path = "./data/duie_schema.json"
    out_path = "./data/id2object.json"
    get_id2object(in_path,out_path)