import json

"""
测试效果的库
"""




"""
功能： 逐文本对比，检查subject的预测效果
params:
"""
def cal_subject_metric(dev_data_file_path,pred_file_path):
    pred_subjects = []
    recall_num = 0
    with open(pred_file_path,'r') as f:    
        cur_subjects = set()
        line =  f.readline()
        while(line):
            if line == "None":
                pred_subjects.append(cur_subjects)
            elif line == '\n':
                recall_num += len(cur_subjects)
                pred_subjects.append(cur_subjects)
                cur_subjects = set()
                line = f.readline()
                continue
            else:
                line = line.strip('\n')
                line = line.split()
                if len(line) > 2:
                    subject = line
                    subject = ' '.join(line[0:-1])
                else:
                    subject = line[0]                
                cur_subjects.add(subject)
            line = f.readline()
    
    gold_subjects= []
    gold_num = 0
    with open(dev_data_file_path,'r') as f:        
        # 为了避免重复，使用set
        line =  f.readline()
        while(line):
            cur_subjects = set() 
            line = json.loads(line) # 以json的方式加载                                    
            spo_list = line['spo_list']
            for spo in spo_list:
                cur_subjects.add(spo['subject'])            
            gold_num += len(cur_subjects)
            gold_subjects.append(cur_subjects)                
            line = f.readline()
    
    
    # 开始计算recall, precision, f1 值
    correct_num = 0
    pred_num = 0
    gold_num = 0
    forget_subject = []  # 找出遗漏的subject
    redundant_subject = [] # 多余的subject
    
    for item in zip(pred_subjects,gold_subjects):        
        pred,gold = item # 每项都是set 
        pred_num += len(pred)
        gold_num += len(gold)
        if len(gold-pred)>0:
            forget_subject.append(gold-pred)
        for subject in pred:
            if subject in gold:
                correct_num+=1
            else:
                redundant_subject.append(subject)

    recall = correct_num / gold_num
    precision = correct_num / recall_num
    if recall+precision == 0: # divide zero error
        f1 = 0
    else:
        f1 = (2*recall*precision) / (recall+precision)
    print(f"correct_num={correct_num}\npred_num={pred_num}\ngold_num={gold_num}\nrecall = {recall}, precision = {precision}, f1 = {f1}")
    print(f"遗漏的subject有：{len(forget_subject)} 个")
    print(f"多余的subject有：{len(redundant_subject)}")
    
    with open('./forget_subject.txt','w') as f:
        for forget in forget_subject:
            f.write(str(forget)+"\n")

    with open('./redundant_subject.txt','w') as f:
        for redundant in redundant_subject:
            f.write(redundant+"\n")
    return (recall,precision,f1)


"""
功能：找出输出和样本之间的关系
"""
def visual_diff(dev_data_file_path,pred_file_path):
    pred_subjects = []
    recall_num = 0
    with open(pred_file_path,'r') as f:    
        cur_subjects = set()
        line =  f.readline()
        while(line):
            if line == "None":
                pred_subjects.append(cur_subjects)
            elif line == '\n':
                recall_num += len(cur_subjects)
                pred_subjects.append(cur_subjects)
                cur_subjects = set()
                line = f.readline()
                continue
            else:
                line = line.strip('\n')
                line = line.split()
                if len(line) > 2:
                    subject = line
                    subject = ' '.join(line[0:-1])
                else:
                    subject = line[0]                
                cur_subjects.add(subject)
            line = f.readline()
    
    gold_subjects= []
    global_subjects_label = {}
    gold_num = 0
    with open(dev_data_file_path,'r') as f:        
        # 为了避免重复，使用set
        line =  f.readline()
        while(line):
            cur_subjects = set()
            cur_subjects_label = []
            line = json.loads(line) # 以json的方式加载                                    
            spo_list = line['spo_list']
            for spo in spo_list:
                cur_subjects.add(spo['subject'])
                global_subjects_label[spo['subject']] = spo['subject_type']
            gold_num += len(cur_subjects)            
            gold_subjects.append(cur_subjects)
            line = f.readline()
        
    # 开始计算recall, precision, f1 值
    correct_num = 0
    pred_num = 0
    gold_num = 0
    forget_subject = []  # 找出遗漏的subject
    redundant_subject = [] # 多余的subject
    forget_label_num = {} # 丢失类别的数目统计
    rebundant_label_num = {} # 冗余类别的数目统计
    cnt = 0
    for item in zip(pred_subjects,gold_subjects):
        pred,gold = item # 每项都是set 
        pred_num += len(pred)
        gold_num += len(gold)
        if len(gold-pred)>0:
            forget_subject.append(gold-pred)
            for forget in (gold-pred):
                if global_subjects_label[forget] not in forget_label_num.keys():
                    forget_label_num[global_subjects_label[forget]] = 1
                else:
                    forget_label_num[global_subjects_label[forget]] += 1
        for subject in pred:
            if subject in gold:
                correct_num+=1
            else:
                redundant_subject.append(subject)
                if subject in global_subjects_label.keys():
                    if global_subjects_label[subject] not in rebundant_label_num.keys():
                        rebundant_label_num[global_subjects_label[subject]] = 0
                    else:
                        rebundant_label_num[global_subjects_label[subject]] += 1
                else:
                    cnt+=1
    recall = correct_num / gold_num
    precision = correct_num / recall_num
    if recall+precision == 0: # divide zero error
        f1 = 0
    else:
        f1 = (2*recall*precision) / (recall+precision)
    print(f"correct_num={correct_num}\npred_num={pred_num}\ngold_num={gold_num}\nrecall = {recall}, precision = {precision}, f1 = {f1}")
    print(f"遗漏的subject有：{len(forget_subject)} 个")
    print(f"多余的subject有：{len(redundant_subject)}")
    
    print(f"遗漏的subject类型统计：{forget_label_num}\n")
    print(f"多余的subject类型统计：{rebundant_label_num}")
    print(f"在训练数据中没有被标签到的：{cnt}")
    return (recall,precision,f1)



if __name__ == "__main__":
    dev_data_file_path = "./data/dev_data.json"
    pred_file_path = "./data/predict/dev_data_subject_predict_model_subject_64236_bert_3.txt"
    #cal_subject_metric(dev_data_file_path,pred_file_path)
    visual_diff(dev_data_file_path,pred_file_path)