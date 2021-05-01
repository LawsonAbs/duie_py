import json
import os
"""
测试效果的库
"""




"""
功能： 逐文本对比，检查subject的预测效果
params:
"""
def cal_subject_metric(dev_data_file_path,pred_file_path):
    print(f"源文件是{dev_data_file_path}，预测文件是{pred_file_path}")
    pred_subjects = []    
    with open(pred_file_path,'r') as f:    
        cur_subjects = set()
        line =  f.readline()
        while(line):
            if line == '\n': # 说明该放入到pred中了
                pred_subjects.append(cur_subjects)
                cur_subjects = set()
                line = f.readline()
                continue
            else:
                line = line.strip('\n')
                line = line.split()
                if len(line) >= 2: # 针对有英文字符出现的情况
                    subject = line
                    subject = ' '.join(line[0:-1])           
                    cur_subjects.add(subject)
                # elif line[0] == 'None': #None 行
                #     cur_subjects.add([])
            line = f.readline()
    
    gold_subjects= []
    gold_num = 0
    gold_subjects_text = [] # 
    with open(dev_data_file_path,'r') as f:        
        # 为了避免重复，使用set
        line =  f.readline()
        while(line):
            cur_subjects = set() 
            line = json.loads(line) # 以json的方式加载
            text = line['text']
            gold_subjects_text.append(text)
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
    forget_subject_text = [] # 遗漏的subject 对应的文本
    for item in zip(pred_subjects,gold_subjects,gold_subjects_text):        
        pred,gold,text = item # 每项都是set 
        pred_num += len(pred)
        gold_num += len(gold)
        if len(gold-pred)>0:
            forget_subject.append(gold-pred)
            forget_subject_text.append(text)
        if len(pred-gold) > 0:
            redundant_subject.append(pred-gold)
        for subject in pred:
            if subject in gold:
                correct_num+=1

    recall = correct_num / gold_num
    precision = correct_num / pred_num
    if recall+precision == 0: # divide zero error
        f1 = 0
    else:
        f1 = (2*recall*precision) / (recall+precision)
    print(f"correct_num={correct_num}\npred_num={pred_num}\ngold_num={gold_num}\nrecall = {recall}, precision = {precision}, f1 = {f1}")
    print(f"遗漏的subject有：{len(forget_subject)} 个")
    print(f"多余的subject有：{len(redundant_subject)}")
    
    forget_file_name = f"{pred_file_path}_f1={f1}_forget_subject.txt"
    redundant_file_name =f"{pred_file_path}_f1={f1}_redundant_subject.txt"
    with open(forget_file_name,'w') as f:
        for forget,text in zip(forget_subject,forget_subject_text):
            f.write(str(forget)+"\t"+text+"\n\n")

    with open(redundant_file_name,'w') as f:
        for redundant in redundant_subject:
            f.write(str(redundant)+"\n")
    return (recall,precision,f1)


"""
功能：找出输出和样本之间的关系
"""
def visual_diff_subject(dev_data_file_path,pred_file_path):
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



# 评测object 的预测性能
def cal_object_metric(pred_file_path,dev_data_file_path):
    gold_objects = []
    with open(dev_data_file_path,'r') as f:        
        line = f.readline()
        while(line):
            line_objects = set() # 当前这条样例中的数据
            line = line.strip("\n")
            line = json.loads(line) # 变成json 数据的格式
            spo_list = line['spo_list']
            for spo in spo_list:
                objects = spo['object']  # dict
                objects_values = list(objects.values())
                if len(objects_values) == 1:
                    line_objects.add(objects_values[0])
                else:
                    for object in objects_values:
                        line_objects.add(object)
            gold_objects.append(line_objects)            
            line = f.readline()
    
    all_objects = []
    with open(pred_file_path,'r') as f:
        line = f.readline()
        while(line):
            line_objects = set() # 当前这条样例中的数据
            line = line.strip("\n")
            line = line.strip('\t')
            if line.startswith("None"):
                all_objects.append([]) # 加入一个空的
            elif len(line) != 0:
                line = line.split("\t")
                all_objects.append(line)
            
            line = f.readline()

    correct_num = 0    
    pred_num = 0
    gold_num = 0
    redundant = []
    forget = []
    for item in zip(all_objects,gold_objects): # 每条样本
        preds,golds = item
        preds = set(preds) # 变成set
        golds = set(golds)
        pred_num += len(preds)
        gold_num += len(golds)
        for pred in preds:
            if pred in golds:
                correct_num+=1
            else:
                redundant.append(pred)                    
        
        for gold in golds:
            if gold not in preds:
                forget.append(gold)
    if gold_num == 0:
        recall = 0
    else:
        recall = correct_num / gold_num
    if pred_num == 0:
        precision = 0
    else:
        precision = correct_num / pred_num
    if (recall + precision) == 0:
        f1 = 0
    else:
        f1 = (2*recall*precision) / (recall+precision)

    forget_temp = pred_file_path+"_forget.txt"
    if os.path.exists(forget_temp):
        os.remove(forget_temp)
    with open(forget_temp,'w') as f:
        for line in forget:                        
            f.write(line+"\n")

    redundant_temp = pred_file_path+"_redundant.txt"
    if os.path.exists(redundant_temp):
        os.remove(redundant_temp)
    with open(redundant_temp,'w') as f:
        for line in redundant:
            f.write(line+"\n")
    #print(f"recall={recall},\nprecistion={precision},\nf1={f1}")
    return (recall,precision,f1)



def cal_subject_object_metric(pred_file_path,dev_data_file_path):
    gold_sub_obj = [] # 里面存储的是一个个的 map
    gold_num = 0
    pred_num = 0
    with open(dev_data_file_path,'r') as f:   
        line = f.readline()
        while(line):
            sub_obj = [] # 当前这条样例中的数据中的subject + object 对
            line = line.strip("\n")
            line = json.loads(line) # 变成json 数据的格式
            spo_list = line['spo_list']
            # print(line)
            # print(type(line))
            for spo in spo_list:
                subject = spo['subject']
                objects = spo['object']  # dict
                objects_values = list(objects.values())                
                
                for object in objects_values:
                    sub_obj.append(f"{subject}_{object}")
                    gold_num += 1    
            gold_sub_obj.append(sub_obj)
            line = f.readline()

    # 将 subject 和 object 组合在一起
    pred_sub_obj = []
    with open(pred_file_path,'r') as f:
        line = f.readline()
        sub_obj = []
        while(line):            
            line = line.strip("\n")
            if len(line) == 0:
                pred_sub_obj.append(sub_obj)
                sub_obj = [] # 重置
                line = f.readline()
                continue
            line = line.split("\t") # tab分割
            if line[0]!='None' and len(line) >= 2:
                sub = line[0]
                obj = line[-1]
                if obj != None:
                    sub_obj.append(sub+"_"+obj)
                    pred_num += 1                
            line = f.readline()

    correct_num = 0
    forget = []
    redundant = []
    for item in zip(pred_sub_obj,gold_sub_obj):
        preds, golds = item
        for pred in  preds:
            if pred in golds:
                correct_num += 1
            else:
                redundant.append(pred)
        for gold in golds:
            if gold not in preds:
                forget.append(gold)

    recall = correct_num/gold_num
    if pred_num == 0:
        precision = 0
    else:
        precision = correct_num/pred_num
    if (recall+precision) == 0:
        f1 = 0
    else:
        f1 = (2*recall*precision) / (recall + precision)
    
    print(f"recall = {recall},\nprecision={precision},\nf1={f1}")
    print(f"correct_num={correct_num}\npred_num={pred_num}\ngold_num={gold_num}")
    forget_temp = pred_file_path+"_forget.txt"
    if os.path.exists(forget_temp):
        os.remove(forget_temp)
    with open(forget_temp,'w') as f:
        for line in forget:
            line = line.split("_")
            left = "".join(line[0:-1])
            right = line[-1]
            f.write(left+"\t"+right+"\n")

    redundant_temp = pred_file_path+"_redundant.txt"
    if os.path.exists(redundant_temp):
        os.remove(redundant_temp)
    with open(redundant_temp,'w') as f:
        for line in redundant:
            line = line.split("_")
            left = "".join(line[0:-1])
            right = line[-1]
            f.write(left+"\t"+right+"\n")    

    return recall,precision,f1

if __name__ == "__main__":
    dev_data_path = "/home/lawson/program/DuIE_py/data/dev_data_5000.json"
    # pred_file_path = "./data/predict/dev_data_subject_predict_model_subject_bert_64236_3333.txt"
    #pred_file_path = "./data/predict/dev_data_subject_predict_model_subject_60000_roberta.txt"
    #cal_subject_metric(dev_data_file_path,pred_file_path)
    #visual_diff_subject(dev_data_file_path,pred_file_path)
    
    pred_file_path = "/home/lawson/program/DuIE_py/data/dev_data_5000_object_predict.txt"
    #cal_subject_object_metric(pred_file_path=pred_file_path,dev_data_file_path= dev_data_path)
    #cal_subject_metric(dev_data_file_path=dev_data_path,pred_file_path=pred_file_path)
    cal_object_metric(pred_file_path,dev_data_path)