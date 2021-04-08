import json

"""
测试效果的库
"""




"""
params:
"""
def cal_subject_metric(dev_data_file_path,pred_file_path):
    pred_subject = []
    recall_num = 0
    with open(pred_file_path,'r') as f:    
        cur_subject = []
        line =  f.readline()
        while(line):
            if line == "None":
                pred_subject.append(cur_subject)                
            elif line == '\n':            
                pred_subject.append(cur_subject)
                cur_subject = []
                line = f.readline()
                continue
            else:                
                line = line.strip('\n')
                line = line.split()
                subject = line[0]
                recall_num += 1
                cur_subject.append(subject)
            line = f.readline()
    
    gold_subjects= []
    gold_num = 0
    with open(dev_data_file_path,'r') as f:        
        cur_subjects = []
        line =  f.readline()
        while(line):
            line = json.loads(line) # 以json的方式加载                                    
            spo_list = line['spo_list']
            for spo in spo_list:
                cur_subjects.append(spo['subject'])
                gold_num += 1            
            gold_subjects.append(cur_subjects)            
            cur_subjects = []            
            line = f.readline()
    
    # 开始计算recall, precision, f1 值
    correct_num = 0
    # 可能不一一对应
    for item in zip(pred_subject,gold_subjects):
        pred,gold = item
        for subject in pred:
            if subject in gold:
                correct_num+=1

    recall = correct_num / gold_num
    precision = correct_num / recall_num
    f1 = (2*recall*precision) / (recall+precision)
    print(f"recall = {recall}, precision = {precision}, f1 = {f1}")
    return (recall,precision,f1)


def cal_subject_metric_2(dev_data_file_path,pred_file_path):
    pred_subject = set()
    recall_num = 0
    with open(pred_file_path,'r') as f:        
        line =  f.readline()
        while(line):
            if line == "None":
                line = f.readline()
                continue
            elif line == '\n':                
                line = f.readline()
                continue
            else:                
                line = line.strip('\n')
                line = line.split()
                subject = line[0]
                recall_num += 1
                pred_subject.add(subject)
            line = f.readline()
    
    gold_subjects= set()
    gold_num = 0
    with open(dev_data_file_path,'r') as f:    
        line =  f.readline()
        while(line):
            line = json.loads(line) # 以json的方式加载                                    
            spo_list = line['spo_list']
            for spo in spo_list:
                gold_subjects.add(spo['subject'])
                gold_num += 1      
            line = f.readline()
    
    # 开始计算recall, precision, f1 值
    correct_num = 0
    # 可能不一一对应
    for pred in zip(pred_subject):        
        if subject in gold_subjects:
            correct_num+=1

    recall = correct_num / gold_num
    precision = correct_num / recall_num
    f1 = (2*recall*precision) / (recall+precision)
    print(f"recall = {recall}, precision = {precision}, f1 = {f1}")

if __name__ == "__main__":
    dev_data_file_path = "./data/dev_data.json"
    pred_file_path = "./data/subject_predict.txt"
    cal_subject_metric(dev_data_file_path,pred_file_path)