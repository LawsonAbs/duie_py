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

if __name__ == "__main__":
    get_subject_class_num(
        train_data_path='./data/dev_data.json'
    )