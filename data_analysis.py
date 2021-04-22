import json
"""
用于分析数据的包
"""


"""
分析数据集中subject的分布情况
"""
def get_subject_class_num(train_data_path):
    subject_class_num = {}
    with open(train_data_path,'r') as f:
        line = f.readline()
        while(line):
            line = json.loads(line)            
            spo_list = line['spo_list']
            for spo in spo_list:
                subject_label = spo['subject_type']
                if subject_label not in subject_class_num.keys():
                    subject_class_num[subject_label] = 1
                else:
                    subject_class_num[subject_label] += 1
            #print(spo_list)
            line = f.readline()
    
    print(subject_class_num)


"""
找出预测结果中 object 不包含@value的部分
"""
def look_error():
    data_path = "/home/lawson/program/DuIE_py/data/test_data_predict.json"
    valid = []
    out_path = "/home/lawson/program/DuIE_py/data/test_data_predict_valid.json"
    cnt = 0
    with open(data_path,'r') as f:
        line = f.readline()        
        while(line):
            line = json.loads(line) # 变成一个json数据
            #print(line)
            spo_list = line['spo_list']
            valid_spo = []
            for spo in spo_list:
                object = spo['object']                
                #print(object)
                if '@value' not in object.keys():
                    cnt +=1
                    #print(line['text'])
                    break
                else:
                    valid_spo.append(spo)
            cur = {}
            cur['text'] = line['text']
            cur['spo_list'] = valid_spo
            valid.append(cur)
            line = f.readline()
    print(cnt)

    with open(out_path,'w') as f:
        for line in valid:
            json.dump(line,f,ensure_ascii=False)
            f.write("\n")
            # f.(str(line)+"\n")
    

def look_error_2():
    data_path = "/home/lawson/program/DuIE_py/data/test_data_predict.json"
    valid = []
    out_path = "/home/lawson/program/DuIE_py/data/test_data_predict_valid.json"
    cnt = 0
    with open(data_path,'r') as f:
        line = f.readline()        
        while(line):
            line = line.strip("\n")
            #line = "{" + line
            line += "}"
            line = json.loads(line) # 变成一个json数据            
            valid.append(line)
            line = f.readline()

    with open(out_path,'w') as f:
        for line in valid:
            json.dump(line,f,ensure_ascii=False)
            f.write("\n")
            # f.(str(line)+"\n")

"""
删除掉 "predicate": "获奖" 的内容
"""
def look_error_3():
    data_path = "/home/lawson/program/DuIE_py/data/dev_data_predict_64236_556706_35000_valid.json"
    valid = []
    out_path = "/home/lawson/program/DuIE_py/data/xx_valid.json"
    cnt = 0
    with open(data_path,'r') as f:
        line = f.readline()        
        while(line):
            line = json.loads(line) # 变成一个json数据
            #print(line)
            spo_list = line['spo_list']
            valid_spo = []
            for spo in spo_list:
                predicate = spo['predicate']
                if predicate == '获奖':
                    cnt +=1                    
                else:
                    valid_spo.append(spo)
            cur = {}
            cur['text'] = line['text']
            cur['spo_list'] = valid_spo
            valid.append(cur)
            line = f.readline()
    print(cnt)

    with open(out_path,'w') as f:
        for line in valid:
            json.dump(line,f,ensure_ascii=False)
            f.write("\n")


# 分析relation 最后的错误点
def analysis_relation(relation_pred_file):
    error_map = {} # 统计错误的类型
    with open(relation_pred_file,'r') as f:
        line = f.readline()
        while(line):
            line = line.strip()
            line = line.split("\t")
            if line[0] == 'x':
                # if len(line) < 5:
                #     line = f.readline()
                #     continue
                if line[4] in error_map.keys():
                    error_map[line[4]] +=1
                else :
                    error_map[line[4]] =0
            line = f.readline()
    res = sorted(error_map.items(), key = lambda d:d[1],reverse=True )
    for i in res:
        print(i)


"""
功能：分析subject 在文本中的位置信息
"""
def analysis_location(file_path):
    location_map = {}
    with open(file_path,'r') as f:
        line = f.readline()
        while(line):
            line = json.loads(line)
            spo_list = line['spo_list']
            text = line['text']
            for spo in spo_list:
                subject = spo['subject']
                index = text.find(subject)
                rate = index/len(text) # 找出一个比例
                rate = round(rate,2) # 保留2 位小数
                #print(rate)
                if rate not in location_map.keys():
                    location_map[rate] = 1
                else:
                    location_map[rate] += 1
            line = f.readline()

    res = sorted(location_map.items(), key = lambda x:x[0], reverse=True )
    for item in res:
        print(item)

if __name__ == "__main__":
    # get_subject_class_num(
    #     train_data_path='./data/dev_data.json'
    # )
    #look_error()
    #analysis_relation(relation_pred_file="/home/lawson/program/DuIE_py/data/predict/relation/relation_predict_513882_roberta.txt")
    analysis_location(file_path='./data/train_data.json')