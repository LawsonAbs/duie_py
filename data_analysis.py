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

if __name__ == "__main__":
    # get_subject_class_num(
    #     train_data_path='./data/dev_data.json'
    # )
    look_error()