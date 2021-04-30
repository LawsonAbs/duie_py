# Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import copy
import codecs
import json
import os
import re
import zipfile
import torch as t
from torch.nn import LogSoftmax
import numpy as np
from torch.nn.modules.activation import Threshold

"""
是否是英文字符或者是连字符
"""
def is_english_char(ch):
    if ('z' >= ch and 'a' <= ch ):
        return True
    if ('Z' >= ch and 'A' <= ch):
        return True
    # if (ch == '-'): 
    #     return True
    return False


def find_entity(text_raw, id_, predictions, tok_to_orig_start_index,
                tok_to_orig_end_index):
    """
    retrieval entity mention under given predicate id for certain prediction.
    this is called by the "decoding" func.
    """
    entity_list = []
    for i in range(len(predictions)):
        if [id_] in predictions[i]:
            j = 0
            while i + j + 1 < len(predictions):
                if [1] in predictions[i + j + 1]:
                    j += 1
                else:
                    break
            entity = ''.join(text_raw[tok_to_orig_start_index[i]:
                                      tok_to_orig_end_index[i + j] + 1])
            entity_list.append(entity)
    return list(set(entity_list))


def decoding(file_path, id2spo, logits_all, seq_len_all,
             tok_to_orig_start_index_all, tok_to_orig_end_index_all):
    """
    model output logits -> formatted spo (as in data set file)
    """
    example_all = []
    with open(file_path, "r", encoding="utf-8") as fp:
        for line in fp:
            example_all.append(json.loads(line))

    formatted_outputs = []
    for (i, (example, logits, seq_len, tok_to_orig_start_index, tok_to_orig_end_index)) in \
            enumerate(zip(example_all, logits_all, seq_len_all, tok_to_orig_start_index_all, tok_to_orig_end_index_all)):

        logits = logits[1:seq_len +
                        1]  # slice between [CLS] and [SEP] to get valid logits
        logits[logits >= 0.5] = 1
        logits[logits < 0.5] = 0
        tok_to_orig_start_index = tok_to_orig_start_index[1:seq_len + 1]
        tok_to_orig_end_index = tok_to_orig_end_index[1:seq_len + 1]
        predictions = []
        for token in logits:
            predictions.append(np.argwhere(token == 1).tolist())

        # format predictions into example-style output
        formatted_instance = {}
        text_raw = example['text']
        complex_relation_label = [8, 10, 26, 32, 46]
        complex_relation_affi_label = [9, 11, 27, 28, 29, 33, 47]

        # flatten predictions then retrival all valid subject id
        flatten_predictions = []
        for layer_1 in predictions:
            for layer_2 in layer_1:
                flatten_predictions.append(layer_2[0])
        subject_id_list = []
        for cls_label in list(set(flatten_predictions)):
            if 1 < cls_label <= 56 and (cls_label + 55) in flatten_predictions:
                subject_id_list.append(cls_label)
        subject_id_list = list(set(subject_id_list))

        # fetch all valid spo by subject id
        spo_list = []
        for id_ in subject_id_list:
            if id_ in complex_relation_affi_label:
                continue  # do this in the next "else" branch
            if id_ not in complex_relation_label:
                subjects = find_entity(text_raw, id_, predictions,
                                       tok_to_orig_start_index,
                                       tok_to_orig_end_index)
                objects = find_entity(text_raw, id_ + 55, predictions,
                                      tok_to_orig_start_index,
                                      tok_to_orig_end_index)
                for subject_ in subjects:
                    for object_ in objects:
                        spo_list.append({
                            "predicate": id2spo['predicate'][id_],
                            "object_type": {
                                '@value': id2spo['object_type'][id_]
                            },
                            'subject_type': id2spo['subject_type'][id_],
                            "object": {
                                '@value': object_
                            },
                            "subject": subject_
                        })
            else:
                #  traverse all complex relation and look through their corresponding affiliated objects
                subjects = find_entity(text_raw, id_, predictions,
                                       tok_to_orig_start_index,
                                       tok_to_orig_end_index)
                objects = find_entity(text_raw, id_ + 55, predictions,
                                      tok_to_orig_start_index,
                                      tok_to_orig_end_index)
                for subject_ in subjects:
                    for object_ in objects:
                        object_dict = {'@value': object_}
                        object_type_dict = {
                            '@value': id2spo['object_type'][id_].split('_')[0]
                        }
                        if id_ in [8, 10, 32, 46
                                   ] and id_ + 1 in subject_id_list:
                            id_affi = id_ + 1
                            object_dict[id2spo['object_type'][id_affi].split(
                                '_')[1]] = find_entity(text_raw, id_affi + 55,
                                                       predictions,
                                                       tok_to_orig_start_index,
                                                       tok_to_orig_end_index)[0]
                            object_type_dict[id2spo['object_type'][
                                id_affi].split('_')[1]] = id2spo['object_type'][
                                    id_affi].split('_')[0]
                        elif id_ == 26:
                            for id_affi in [27, 28, 29]:
                                if id_affi in subject_id_list:
                                    object_dict[id2spo['object_type'][id_affi].split('_')[1]] = \
                                    find_entity(text_raw, id_affi + 55, predictions, tok_to_orig_start_index, tok_to_orig_end_index)[0]
                                    object_type_dict[id2spo['object_type'][id_affi].split('_')[1]] = \
                                    id2spo['object_type'][id_affi].split('_')[0]
                        spo_list.append({
                            "predicate": id2spo['predicate'][id_],
                            "object_type": object_type_dict,
                            "subject_type": id2spo['subject_type'][id_],
                            "object": object_dict,
                            "subject": subject_
                        })

        formatted_instance['text'] = example['text']
        formatted_instance['spo_list'] = spo_list
        formatted_outputs.append(formatted_instance)
    return formatted_outputs


def write_prediction_results(formatted_outputs, file_path):
    """write the prediction results"""

    with codecs.open(file_path, 'w', 'utf-8') as f:
        for formatted_instance in formatted_outputs:
            json_str = json.dumps(formatted_instance, ensure_ascii=False)
            f.write(json_str)
            f.write('\n')
        zipfile_path = file_path + '.zip'
        f = zipfile.ZipFile(zipfile_path, 'w', zipfile.ZIP_DEFLATED)
        f.write(file_path)

    return zipfile_path


def get_precision_recall_f1(golden_file, predict_file):
    r = os.popen(
        'python3 ./re_official_evaluation.py --golden_file={} --predict_file={}'.
        format(golden_file, predict_file))
    result = r.read()  # 这是将比较结果写入到了result 中
    r.close()
    precision = float(
        re.search("\"precision\", \"value\":.*?}", result).group(0).lstrip(
            "\"precision\", \"value\":").rstrip("}"))
    recall = float(
        re.search("\"recall\", \"value\":.*?}", result).group(0).lstrip(
            "\"recall\", \"value\":").rstrip("}"))
    f1 = float(
        re.search("\"f1-score\", \"value\":.*?}", result).group(0).lstrip(
            "\"f1-score\", \"value\":").rstrip("}"))

    return precision, recall, f1


"""
功能：由 预测subject的label 得到实体
01.适用的情况是 不同的B标签，统一的I/O标签，
02.params:
 origin_text:原文本，用于消除不认识的字变成 [UNK] 的问题
03.仅预测argmax
"""
def decode_subject_bp(logits,id2subject_map,input_ids,tokenizer,batch_origin_info,batch_offset_mapping):
    m = LogSoftmax(dim=-1)
    a = m(logits)
    batch_indexs = a.argmax(-1) # 在该维度找出值最大的，就是预测出来的标签    
    batch_subjects =[]
    batch_labels = []    
    for i,indexs in enumerate(batch_indexs): # 找出index 
        offset_mapping  = batch_offset_mapping[i] # 拿到当前的offset_mapping
        origin_info = batch_origin_info[i]
        origin_text = origin_info['text']
        tokens = tokenizer.convert_ids_to_tokens(input_ids[i]) # 得到原字符串
        subjects = [] # 预测出最后的结果
        labels = []
        cur_subject = ""
        for j,ind in enumerate(indexs):
            if ind > 1 : # 说明是一个标签的开始
                offset = offset_mapping[j]
                left,right = tuple(offset)
                # 判断上一个字符是否是字母结束（英文）/数字并且当前的是否不是#开头 => 需要安排一个空格
                if (cur_subject!="" 
                    and ( is_english_char(cur_subject[-1]) or ('9'>= cur_subject[-1] and '0'<= cur_subject[-1]))
                    and not(tokens[j].startswith("#"))
                    and is_english_char(origin_text[left]) # 如果其后也是英文 
                    ):
                    cur_subject+=" "
                cur_subject += origin_text[left:right]
                #cur_subject += origin_text[i-1] # 因为有的字无法识别，所以这里用origin_text. i-1 是因为 相比而言，origin_text 少了 [CLS]
                cur_subject_label = id2subject_map[str(ind.item())]
            if ind == 1 and cur_subject!="": # 说明是中间部分，且 cur_subject 不为空
                offset = offset_mapping[j]
                left,right = tuple(offset)
                if( 
                    (is_english_char(cur_subject[-1]) or ('9'>= cur_subject[-1] and '0'<= cur_subject[-1]))
                    and not(tokens[j].startswith("#"))
                    and is_english_char(origin_text[left]) # 如果其后也是英文
                    ):
                    cur_subject+=" "
                cur_subject += origin_text[left:right]
            elif ind == 0 and cur_subject!="": # 将 cur_subject 放入到 subjects 中
                cur_subject = cur_subject.replace("#","") # 替换掉，因为这会干扰后面的实现
                # 后处理部分之删除不符合规则的数据
                # 后处理部分之判断subject 的长度：如果长度大于1的才放进去
                if (not is_year_month_day(cur_subject) 
                    and (len(cur_subject)> 1) # 如果subject 的长度大于1
                    and cur_subject_label != 19 # 如果不是第19 类（杂类），那么就放入其中
                    ):
                    subjects.append(cur_subject)                                
                    cur_subject_label = cur_subject_label.replace("#","")
                    labels.append(cur_subject_label)
                cur_subject = ""
        
        batch_subjects.append(subjects)
        batch_labels.append(labels)
    # 然后再找出对应的内容    
    return batch_subjects,batch_labels




'''
将最大和次大两种都放入到模型结果
'''
def decode_subject(logits,id2subject_map,input_ids,tokenizer,batch_origin_info,batch_offset_mapping,all_known_subjects):
    # step1.对top1 进行解码
    m = LogSoftmax(dim=-1)
    a = m(logits)
    first_batch_indexs = a.argmax(-1) # 在该维度找出值最大的，就是预测出来的标签    

    # step 2.对top2进行解码
    c = t.topk(logits,k=2,dim=-1) # 直接在logits上取
    value,batch_index = c
    # 将index翻转，取翻转后的第二行就是次大的index 
    batch_index = batch_index.transpose(2,1)
    first_batch_indexs = batch_index[:,0,:]
    second_batch_indexs = batch_index[:,1,:]
    # [batch_size,max_seq_length]

    batch_value = value.transpose(2,1)
    first_batch_values = batch_value[:,0,:]
    second_batch_values = batch_value[:,1,:]
    up_threshold = 7 # 间外
    down_threshold = 4 # 间内
    val_threshold = 4 # 单个值也必须满足一定的要求
    # step3.开始解码
    batch_subjects =[]
    batch_labels = []
    i = 0
    for first_indexs,first_values,second_indexs,second_values in zip(first_batch_indexs,first_batch_values,second_batch_indexs,second_batch_values): # 找出index 
        offset_mapping  = batch_offset_mapping[i] # 拿到当前的offset_mapping
        origin_info = batch_origin_info[i]
        origin_text = origin_info['text']
        tokens = tokenizer.convert_ids_to_tokens(input_ids[i]) # 得到原字符串
        subjects = [] # 预测出最后的结果
        labels = []
        cur_subject = ""
        for j,ind in enumerate(first_indexs):
            if ind > 1 : # 说明是一个标签的开始
                offset = offset_mapping[j]
                left,right = tuple(offset)
                # 判断上一个字符是否是字母结束（英文）/数字并且当前的是否不是#开头 => 需要安排一个空格
                if (cur_subject!="" 
                    and ( is_english_char(cur_subject[-1]) or ('9'>= cur_subject[-1] and '0'<= cur_subject[-1]))
                    and not(tokens[j].startswith("#"))
                    and is_english_char(origin_text[left]) # 如果其后也是英文 
                    and origin_text[left] != '-' # 不是连字符
                    ):
                    cur_subject+=" "
                cur_subject += origin_text[left:right]
                
                #cur_subject += origin_text[i-1] # 因为有的字无法识别，所以这里用origin_text. i-1 是因为 相比而言，origin_text 少了 [CLS]
                cur_subject_label = id2subject_map[str(ind.item())]
            if ind == 1 and cur_subject!="": # 说明是中间部分，且 cur_subject 不为空
                offset = offset_mapping[j]
                left,right = tuple(offset)
                if( 
                    (is_english_char(cur_subject[-1]) or ('9'>= cur_subject[-1] and '0'<= cur_subject[-1]))
                    and not(tokens[j].startswith("#"))
                    and is_english_char(origin_text[left]) # 如果其后也是英文
                    and origin_text[left] !='-'
                    ):
                    cur_subject+=" "
                cur_subject += origin_text[left:right]
            elif ind == 0 and cur_subject!="": # 将 cur_subject 放入到 subjects 中
                cur_subject = cur_subject.replace("#","") # 替换掉，因为这会干扰后面的实现
                # 后处理部分之删除不符合规则的数据
                # 后处理部分之判断subject 的长度：如果长度大于1的才放进去
                if (not is_year_month_day(cur_subject) 
                    and (len(cur_subject)> 1)
                    and cur_subject_label != 19 # 如果不是第19 类（杂类），那么就放入其中
                    ):
                    subjects.append(cur_subject)                                
                    cur_subject_label = cur_subject_label.replace("#","")
                    labels.append(cur_subject_label)
                cur_subject = ""

        # 后处理之添加书名号中的内容
        target = addBookName(origin_text)
        for word in target:
            if word not in subjects:
                subjects.append(word)
                labels.append("后处理1")

        # flag = 1
        # for know in all_known_subjects:
        #     cur_len = len(origin_text)
        #     if know in origin_text:
        #         if ((origin_text.find(know) /cur_len) < 0.1
        #             and know not in subjects
        #             ):
        #             for subject in subjects: # 判断是否已经有这个作为开头了，如果有的话，则不在放入
        #                 if subject.startswith(know) or (subject.find(know)!=-1):
        #                     flag = 0
        #             if flag:
        #                 subjects.append(know)
        #                 labels.append('后处理2')
        
        detail_info = [(fi.item(),round(fv.item(),3),si.item(),round(sv.item(),3),t) for fi,fv,si,sv,t in zip(first_indexs,first_values,second_indexs,second_values,tokens)]


        # 从次大的下标中寻找，此时需要注意其得分情况
        cur_subject = ""
        k = 0
        pre_distance = 0 # 表示的是上一次fi_val 和 se_val 之间的距离差值
        cur_distance = 0 # 当前两者的差值
        for fi_ind,fi_val,se_ind,se_val in zip(first_indexs,first_values,second_indexs,second_values):
            cur_distance = abs(se_val-fi_val)

            # 第一种结束方式——说明是分界点
            if (abs(cur_distance-pre_distance) > up_threshold 
                and k # 排除掉CLS向量
                ): 
                if cur_subject!="":
                    cur_subject = cur_subject.replace("#","") # 替换掉，因为这会干扰后面的实现
                    flag = 1
                    cur_subject = cur_subject.strip("《》，。+-.:：（）()、/\\！!") # 剃掉两边的所有符号                        
                    for have in subjects: # 判断当前此轮预测的结果是否出现在之前的预测结果中
                        if (have.startswith(cur_subject)
                            or have.find(cur_subject)!=-1 # 如果该串作为子串出现过
                            ):
                            flag = 0
                            break
                    # 后处理部分之删除不符合规则的数据
                    # 后处理部分之判断subject 的长度：如果长度大于1的才放进去
                    if (not is_year_month_day(cur_subject)
                        and cur_subject_label != 19 # 如果不是第19 类（杂类），那么就放入其中
                        and len(cur_subject) > 1
                        and flag
                        ):                        
                        subjects.append(cur_subject)
                        cur_subject_label = "top_2"
                        labels.append(cur_subject_label)
                    cur_subject = ""
                else: # 说明是一个标签的开始
                    if (fi_ind<=1 # 如果它在top_1 中就已经是标签了，则要过滤
                        and cur_distance < pre_distance
                        and se_val > 3
                        ): 
                        offset = offset_mapping[k]
                        left,right = tuple(offset)
                        # 判断上一个字符是否是字母结束（英文）/数字并且当前的是否不是#开头 => 需要安排一个空格
                        if (cur_subject!="" 
                            and ( is_english_char(cur_subject[-1]) or ('9'>= cur_subject[-1] and '0'<= cur_subject[-1]))
                            and not(tokens[k].startswith("#"))
                            and is_english_char(origin_text[left]) # 如果其后也是英文 
                            and origin_text[left] != '-' # 不是连字符
                            ):
                            cur_subject+=" "
                        cur_subject += origin_text[left:right]            
                        cur_subject_label = id2subject_map[str(se_ind.item())]
                    else:
                        cur_subject = ""
            
            # 说明是中间部分
            elif (cur_subject!="" 
                and abs(cur_distance - pre_distance) < down_threshold                
                ): 
                if( se_val > val_threshold# 当前也要过关
                    and fi_ind <= 1
                    ): 
                    offset = offset_mapping[k]
                    left,right = tuple(offset)
                    if( 
                        (is_english_char(cur_subject[-1]) or ('9'>= cur_subject[-1] and '0'<= cur_subject[-1]))
                        and not(tokens[k].startswith("#"))
                        and is_english_char(origin_text[left]) # 如果其后也是英文
                        and origin_text[left] != '-'
                        ):
                        cur_subject+=" "
                    cur_subject += origin_text[left:right]
                else:
                    cur_subject = ""
            
            # 第二种结束标志
            # 结束的标志要严格， 所以这里用or，两者
            elif(abs(cur_distance-pre_distance) > down_threshold  # 超过间内的距离
                or se_val < val_threshold
                ): 
                if cur_subject!="":
                    cur_subject = cur_subject.replace("#","") # 替换掉，因为这会干扰后面的实现
                    flag = 1
                    cur_subject = cur_subject.strip("《》，。+-.:：（）()、/\\！!") # 剃掉两边的所有符号                        
                    for have in subjects: # 判断当前此轮预测的结果是否出现在之前的预测结果中
                        if (have.startswith(cur_subject)
                            or have.find(cur_subject)!=-1 # 如果该串作为子串出现过
                            ):
                            flag = 0
                            break
                    # 后处理部分之删除不符合规则的数据
                    # 后处理部分之判断subject 的长度：如果长度大于1的才放进去
                    if (not is_year_month_day(cur_subject)
                        and cur_subject_label != 19 # 如果不是第19 类（杂类），那么就放入其中
                        and len(cur_subject) > 1
                        and flag
                        ):                      
                        subjects.append(cur_subject)
                        cur_subject_label = "top_2"
                        labels.append(cur_subject_label)
                    cur_subject = ""
            k+=1
            pre_distance = cur_distance

        batch_subjects.append(subjects)
        batch_labels.append(labels)
        i+=1 

    return batch_subjects,batch_labels




def decode_subject_2(logits,id2subject_map,input_ids,tokenizer,batch_origin_info,batch_offset_mapping):
    m = LogSoftmax(dim=-1)
    a = m(logits)
    batch_indexs = a.argmax(-1) # 在该维度找出值最大的，就是预测出来的标签    
    batch_subjects =[]
    batch_labels = []    
    for i,indexs in enumerate(batch_indexs): # 找出index 
        offset_mapping  = batch_offset_mapping[i] # 拿到当前的offset_mapping
        origin_info = batch_origin_info[i]
        origin_text = origin_info['text']
        tokens = tokenizer.convert_ids_to_tokens(input_ids[i]) # 得到原字符串
        subjects = [] # 预测出最后的结果
        labels = []
        cur_subject = ""
        for j,ind in enumerate(indexs):            
            if ind == 1 and cur_subject!="": # 说明是中间部分，且 cur_subject 不为空
                offset = offset_mapping[j]
                left,right = tuple(offset)
                if( 
                    (is_english_char(cur_subject[-1]) or ('9'>= cur_subject[-1] and '0'<= cur_subject[-1]))
                    and not(tokens[j].startswith("#"))
                    and is_english_char(origin_text[left]) # 如果其后也是英文
                    ):
                    cur_subject+=" "
                cur_subject += origin_text[left:right]
            elif ind == 0 and cur_subject!="": # 将 cur_subject 放入到 subjects 中
                cur_subject = cur_subject.replace("#","") # 替换掉，因为这会干扰后面的实现
                # 后处理部分之删除不符合规则的数据
                # 后处理部分之判断subject 的长度：如果长度大于1的才放进去
                if (not is_year_month_day(cur_subject) 
                    and (len(cur_subject)> 1)                    
                    ):
                    subjects.append(cur_subject)                                                    
                    labels.append("I")
                cur_subject = ""            
            if ind == 1 and cur_subject=="": # 开头
                offset = offset_mapping[j]
                left,right = tuple(offset)                
                cur_subject += origin_text[left:right]
        
        batch_subjects.append(subjects)
        batch_labels.append(labels)
    # 然后再找出对应的内容    
    return batch_subjects,batch_labels

'''
使用完整的BIO标签做
'''
def decode_subject_3(logits,id2subject_map,input_ids,tokenizer,batch_origin_info,batch_offset_mapping):
    m = LogSoftmax(dim=-1)
    a = m(logits)
    batch_indexs = a.argmax(-1) # 在该维度找出值最大的，就是预测出来的标签    
    batch_subjects =[]
    batch_labels = []    
    for i,indexs in enumerate(batch_indexs): # 找出index 
        offset_mapping  = batch_offset_mapping[i] # 拿到当前的offset_mapping
        origin_info = batch_origin_info[i]
        origin_text = origin_info['text']
        tokens = tokenizer.convert_ids_to_tokens(input_ids[i]) # 得到原字符串
        subjects = [] # 预测出最后的结果
        labels = []
        cur_subject = ""
        cur_label_id = -100 # 赋初值
        for j,ind in enumerate(indexs):
            if ind %2 == 1: # B标签
                if cur_subject == "" : # 是B且是开头
                    cur_label_id = ind.item() # 拿到标签的id
                    offset = offset_mapping[j]
                    left,right = tuple(offset)
                    # 判断上一个字符是否是字母结束（英文）/数字并且当前的是否不是#开头 => 需要安排一个空格
                    if (cur_subject!="" 
                        and ( is_english_char(cur_subject[-1]) or ('9'>= cur_subject[-1] and '0'<= cur_subject[-1]))
                        and not(tokens[j].startswith("#"))
                        and is_english_char(origin_text[left]) # 如果其后也是英文 
                        ):
                        cur_subject+=" "
                    cur_subject += origin_text[left:right]
                    cur_subject_label = id2subject_map[str(cur_label_id)]
                    
                elif cur_subject != "": # 是另一组B的开始
                    # 先放之前的内容
                    cur_subject = cur_subject.replace("#","")
                    # 后处理部分之删除不符合规则的数据                    
                    if (not is_year_month_day(cur_subject) ):
                        subjects.append(cur_subject)                        
                        labels.append(cur_subject_label)                    
                    cur_subject= ""


                    # 接着安置当下的内容
                    offset = offset_mapping[j]
                    left,right = tuple(offset)
                    cur_subject += origin_text[left:right]
                    cur_label_id = ind.item() # 拿到标签的id
                    cur_subject_label = id2subject_map[str(cur_label_id)] # 拿到标签

            elif ind %2 ==0 and ind : # I标签
                if ind.item() == cur_label_id + 1: # 如果能够和上一个B匹配
                    cur_subject = cur_subject.replace("#","") # 替换掉，因为这会干扰后面的实现
                    offset = offset_mapping[j]
                    left,right = tuple(offset)
                    # 判断上一个字符是否是字母结束（英文）/数字并且当前的是否不是#开头 => 需要安排一个空格
                    if (cur_subject!="" 
                        and ( is_english_char(cur_subject[-1]) or ('9'>= cur_subject[-1] and '0'<= cur_subject[-1]))
                        and not(tokens[j].startswith("#"))
                        and is_english_char(origin_text[left]) # 如果其后也是英文 
                        ):
                        cur_subject+=" "
                    cur_subject += origin_text[left:right]

                elif cur_subject!="":# 如果不能匹配且之前有内容
                    cur_subject = cur_subject.replace("#","")                    
                    # 后处理部分之删除不符合规则的数据                    
                    if (not is_year_month_day(cur_subject) ):
                        subjects.append(cur_subject)                        
                        labels.append(cur_subject_label)
                    cur_subject= ""
                    cur_label_id = -100 # 也得重置

            elif ind == 0 and cur_subject!='': # O标签
                cur_subject = cur_subject.replace("#","")                    
                # 后处理部分之删除不符合规则的数据
                # 01.不是年月日
                if (not is_year_month_day(cur_subject) ):
                    subjects.append(cur_subject)                        
                    labels.append(cur_subject_label)
                cur_subject= ""
                cur_label_id = -100

        batch_subjects.append(subjects)
        batch_labels.append(labels)
    # 然后再找出对应的内容    
    return batch_subjects,batch_labels



def decode_subject_crf(batch_logits,id2subject_map,input_ids,tokenizer,batch_origin_info,batch_offset_mapping):            
    batch_subjects =[]
    batch_labels = []    
    for i,indexs in enumerate(batch_logits): # 找出index 
        offset_mapping  = batch_offset_mapping[i] # 拿到当前的offset_mapping
        origin_info = batch_origin_info[i]
        origin_text = origin_info['text']
        tokens = tokenizer.convert_ids_to_tokens(input_ids[i]) # 得到原字符串
        subjects = [] # 预测出最后的结果
        labels = []
        cur_subject = ""
        cur_label_id = -100 # 赋初值
        for j,ind in enumerate(indexs):
            if ind %2 == 1: # B标签
                if cur_subject == "" : # 是B且是开头
                    cur_label_id = ind.item() # 拿到标签的id
                    offset = offset_mapping[j]
                    left,right = tuple(offset)
                    # 判断上一个字符是否是字母结束（英文）/数字并且当前的是否不是#开头 => 需要安排一个空格
                    if (cur_subject!="" 
                        and ( is_english_char(cur_subject[-1]) or ('9'>= cur_subject[-1] and '0'<= cur_subject[-1]))
                        and not(tokens[j].startswith("#"))
                        and is_english_char(origin_text[left]) # 如果其后也是英文 
                        ):
                        cur_subject+=" "
                    cur_subject += origin_text[left:right]
                    cur_subject_label = id2subject_map[str(cur_label_id)]
                    
                elif cur_subject != "": # 是另一组B的开始
                    # 先放之前的内容
                    cur_subject = cur_subject.replace("#","")
                    # 后处理部分之删除不符合规则的数据                    
                    if (not is_year_month_day(cur_subject) ):
                        subjects.append(cur_subject)                        
                        labels.append(cur_subject_label)                    
                    cur_subject= ""


                    # 接着安置当下的内容
                    offset = offset_mapping[j]
                    left,right = tuple(offset)
                    cur_subject += origin_text[left:right]
                    cur_label_id = ind.item() # 拿到标签的id
                    cur_subject_label = id2subject_map[str(cur_label_id)] # 拿到标签

            elif ind %2 ==0 and ind : # I标签
                if ind.item() == cur_label_id + 1: # 如果能够和上一个B匹配
                    cur_subject = cur_subject.replace("#","") # 替换掉，因为这会干扰后面的实现
                    offset = offset_mapping[j]
                    left,right = tuple(offset)
                    # 判断上一个字符是否是字母结束（英文）/数字并且当前的是否不是#开头 => 需要安排一个空格
                    if (cur_subject!="" 
                        and ( is_english_char(cur_subject[-1]) or ('9'>= cur_subject[-1] and '0'<= cur_subject[-1]))
                        and not(tokens[j].startswith("#"))
                        and is_english_char(origin_text[left]) # 如果其后也是英文 
                        ):
                        cur_subject+=" "
                    cur_subject += origin_text[left:right]

                elif cur_subject!="":# 如果不能匹配且之前有内容
                    cur_subject = cur_subject.replace("#","")                    
                    # 后处理部分之删除不符合规则的数据                    
                    if (not is_year_month_day(cur_subject) ):
                        subjects.append(cur_subject)                        
                        labels.append(cur_subject_label)                    
                    cur_subject= ""
                    cur_label_id = -100 # 也得重置

            elif ind == 0 and cur_subject!='': # O标签
                cur_subject = cur_subject.replace("#","")                    
                # 后处理部分之删除不符合规则的数据
                # 01.不是年月日
                if (not is_year_month_day(cur_subject) ):
                    subjects.append(cur_subject)                        
                    labels.append(cur_subject_label)
                cur_subject= ""
                cur_label_id = -100

        batch_subjects.append(subjects)
        batch_labels.append(labels)
    # 然后再找出对应的内容    
    return batch_subjects,batch_labels



'''
使用start-end 的方式标注
'''
def decode_subject_4(batch_logits,input_ids,tokenizer,batch_origin_info,batch_offset_mapping):            
    m = LogSoftmax(dim=-1)
    a = m(batch_logits)
    batch_indexs = a.argmax(-1) # 在该维度找出值最大的，就是预测出来的标签
    batch_subjects =[]
    for i,indexs in enumerate(batch_indexs): # 找出index 
        offset_mapping  = batch_offset_mapping[i] # 拿到当前的offset_mapping
        origin_info = batch_origin_info[i]
        origin_text = origin_info['text']
        tokens = tokenizer.convert_ids_to_tokens(input_ids[i]) # 得到原字符串
        subjects = [] # 预测出最后的结果        
        cur_subject = ""
        
        for j,ind in enumerate(indexs):
            if ind == 1: # start or end
                if cur_subject == "" : # 是开头   
                    offset = offset_mapping[j]
                    left,right = tuple(offset)
                    # 判断上一个字符是否是字母结束（英文）/数字并且当前的是否不是#开头 => 需要安排一个空格
                    if (cur_subject!="" 
                        and ( is_english_char(cur_subject[-1]) or ('9'>= cur_subject[-1] and '0'<= cur_subject[-1]))
                        and not(tokens[j].startswith("#"))
                        and is_english_char(origin_text[left]) # 如果其后也是英文 
                        ):
                        cur_subject+=" "
                    cur_subject += origin_text[left:right]                    
                    
                elif cur_subject != "":
                    # 先把之前的内容存下来
                    offset = offset_mapping[j]
                    left,right = tuple(offset)
                    # 判断上一个字符是否是字母结束（英文）/数字并且当前的是否不是#开头 => 需要安排一个空格
                    if (cur_subject!="" 
                        and ( is_english_char(cur_subject[-1]) or ('9'>= cur_subject[-1] and '0'<= cur_subject[-1]))
                        and not(tokens[j].startswith("#"))
                        and is_english_char(origin_text[left]) # 如果其后也是英文 
                        ):
                        cur_subject+=" "
                    cur_subject += origin_text[left:right]
                                    
                    cur_subject = cur_subject.replace("#","")
                    # 后处理部分之删除不符合规则的数据                    
                    if (not is_year_month_day(cur_subject) ):
                        subjects.append(cur_subject)                                     
                    cur_subject= "" # 重置

            elif ind == 0 : 
                if cur_subject!="" : # 说明是中间的部分
                    offset = offset_mapping[j]
                    left,right = tuple(offset)
                    # 判断上一个字符是否是字母结束（英文）/数字并且当前的是否不是#开头 => 需要安排一个空格
                    if (cur_subject!="" 
                        and ( is_english_char(cur_subject[-1]) or ('9'>= cur_subject[-1] and '0'<= cur_subject[-1]))
                        and not(tokens[j].startswith("#"))
                        and is_english_char(origin_text[left]) # 如果其后也是英文 
                        ):
                        cur_subject+=" "
                    cur_subject += origin_text[left:right]
                        

        batch_subjects.append(subjects)
    # 然后再找出对应的内容    
    return batch_subjects



"""
功能：由 预测object 的labels 得到object，
params:
 logits: 预测的值，需要经过softmax处理，然后得到结果
 id2object_map
 ...
 object_origin_info: 用于帮助恢复UNK 字

01.这些参数都是batch 级别的
"""
def decode_object_bp(logits,id2object_map,tokenizer,batch_object_input_ids,batch_object_origin_info,batch_object_offset_mapping):
    m = LogSoftmax(dim=-1)
    a = m(logits)
    batch_indexs = a.argmax(-1) # 在该维度找出值最大的，就是预测出来的标签
    batch_objects = [] # 预测出最后的结果
    batch_labels = []    

    # step 2.对top2进行解码
    c = t.topk(logits,k=2,dim=-1) # 直接在logits上取
    value,batch_index = c
    # 将index翻转，取翻转后的第二行就是次大的index 
    batch_index = batch_index.transpose(2,1)
    second_batch_indexs = batch_index[:,1,:]
    # [batch_size,max_seq_length]

    batch_value = value.transpose(2,1)
    second_batch_values = batch_value[:,1,:]
    threshold = 10

    for item in zip(batch_indexs,batch_object_input_ids,batch_object_offset_mapping,batch_object_origin_info,second_batch_indexs,second_batch_values):
        first_indexs,input_ids,offset , origin_info,second_indexs,second_values = item
        text_raw = origin_info['text']
        tokens = tokenizer.convert_ids_to_tokens(input_ids) # 得到原字符串
        objects = []
        labels = []
        cur_object = ""
        for i,ind in enumerate(first_indexs):
            if ind > 1 : # 说明是一个标签的开始                
                left,right = tuple(offset[i])
                # 判断上一个字符是否是字母结束（英文）/数字并且当前的是否不是#开头 => 需要安排一个空格
                if (cur_object!="" 
                    and ( is_english_char(cur_object[-1]) or ('9'>= cur_object[-1] and '0'<= cur_object[-1]))
                    and not(tokens[i].startswith("#"))
                    and is_english_char(text_raw[left]) # 如果其后也是英文
                    and text_raw[left] !='-'
                    ):
                    cur_object+=" "
                cur_object+= text_raw[left:right]
                cur_object_label = id2object_map[str(ind.item())]
            if ind == 1 and cur_object!="": # 说明是中间部分，且 cur_subject 不为空
                left,right = tuple(offset[i])
                if (cur_object!="" 
                    and ( is_english_char(cur_object[-1]) or ('9'>= cur_object[-1] and '0'<= cur_object[-1]))
                    and not(tokens[i].startswith("#"))
                    and is_english_char(text_raw[left]) # 如果其后也是英文 
                    and text_raw[left] !='-'
                    ):
                    cur_object+=" "
                cur_object+= text_raw[left:right] 
            elif (ind == 0 
                    and cur_object!=""                    
                    ): # 将 cur_subject 放入到 subjects 中            
                    if len(cur_object) > 1:
                        cur_object = cur_object.replace("#","")                
                        objects.append(cur_object)
                        labels.append(cur_object_label)
                    cur_object = ""
        
        
        # # 从次大的下标中寻找，此时只需要注意其得分情况
        cur_object = ""
        k = 0
        for ind,val in zip(second_indexs,second_values):
            if val > threshold: # 说明是一个标签的开始
                left,right = tuple(offset[k])
                # 判断上一个字符是否是字母结束（英文）/数字并且当前的是否不是#开头 => 需要安排一个空格
                if (cur_object!="" 
                    and ( is_english_char(cur_object[-1]) or ('9'>= cur_object[-1] and '0'<= cur_object[-1]))
                    and not(tokens[k].startswith("#"))
                    and is_english_char(text_raw[left]) # 如果其后也是英文 
                    and text_raw[left] != '-' # 不是连字符
                    ):
                    cur_object+=" "
                cur_object += text_raw[left:right]
                cur_object_label = id2object_map[str(ind.item())]
            elif cur_object!="" and val > threshold: # 说明是中间部分，且 cur_subject 不为空                
                left,right = tuple(offset[k])
                if( 
                    (is_english_char(cur_object[-1]) or ('9'>= cur_object[-1] and '0'<= cur_object[-1]))
                    and not(tokens[k].startswith("#"))
                    and is_english_char(text_raw[left]) # 如果其后也是英文
                    and text_raw[left] != '-'
                    ):
                    cur_object+=" "
                cur_object += text_raw[left:right]
            
            
            # 注意这里放入的条件
            elif val < threshold :
                if cur_object!="" :
                    cur_object = cur_object.replace("#","") # 替换掉，因为这会干扰后面的实现
                    cur_object = cur_object.strip("《》，。+-.:：（）()、/\\！!") # 剃掉两边的所有符号
                    flag = 1
                    for have in objects: 
                        # 判断当前此轮预测的结果是否出现在之前的预测结果中
                        # 或者是否是其子串
                        if have.startswith(cur_object) or (have.find(cur_object)!=-1):
                            flag = 0
                            break

                    # 后处理部分之删除不符合规则的数据
                    # 后处理部分之判断subject 的长度：如果长度大于1的才放进去
                    if (not is_year_month_day(cur_object)                     
                        and cur_object_label != 19 # 如果不是第19 类（杂类），那么就放入其中
                        and len(cur_object) > 1
                        and flag
                        ):
                        objects.append(cur_object)
                        cur_object_label = "top_2"
                        labels.append(cur_object_label)
                cur_object = "" # 只要小于阈值，就该重置
            k+=1
        
        batch_objects.append(objects)
        batch_labels.append(labels)
    # 然后再找出对应的内容
    return batch_objects,batch_labels



def decode_object(logits,id2object_map,tokenizer,batch_object_input_ids,
                batch_object_origin_info,batch_object_offset_mapping,logger):
# step1.对top1 进行解码
    m = LogSoftmax(dim=-1)
    a = m(logits)
    first_batch_indexs = a.argmax(-1) # 在该维度找出值最大的，就是预测出来的标签        
    country = ['秦','晋','唐','宋','元','明','清','汉','吴','隋','蜀']
    number = ['1','2','3','4','5','6','7','8','9','一','二','三','肆','伍','陆','柒','捌','玖','拾']

    # step 2.对top2进行解码
    c = t.topk(logits,k=2,dim=-1) # 直接在logits上取
    value,batch_index = c
    # 将index翻转，取翻转后的第二行就是次大的index 
    batch_index = batch_index.transpose(2,1)
    first_batch_indexs = batch_index[:,0,:]
    second_batch_indexs = batch_index[:,1,:]
    # [batch_size,max_seq_length]

    batch_value = value.transpose(2,1)
    first_batch_values = batch_value[:,0,:]
    second_batch_values = batch_value[:,1,:]

    # 对 first_batch_indexs 的值进行一个预处理（矫正），用于找出被漏掉的大概率数据
    for batch_index,origin in zip(first_batch_indexs,batch_object_origin_info):
        for i in range(len(batch_index)):
            fir_val = batch_index[i]
            if fir_val == 0:
                # 矫正 1 1 0 1 1 1 0 这种情况
                if ( (i+2) < len(batch_index) and (i-2) >= 0
                    and batch_index[i+1] == 1 
                    and batch_index[i+2] == 1
                    and batch_index[i-1] == 1
                    and batch_index[i-2] >= 1
                    ):
                    batch_index[i] = 1                     
                    logger.info(f"被修改的文本是：{origin}")
                # 矫正 0 1 1 1 0 0 这种情况
                elif ( (i+2) < len(batch_index)
                    and batch_index[i+1] == 1 
                    and batch_index[i+2] == 1
                    ):
                    batch_index[i] = 2
                    logger.info(f"被修改的文本是：{origin}")
                
    # step3.开始解码
    batch_objects =[]
    batch_labels = []
    i = 0
    for first_indexs,first_values,second_indexs,second_values in zip(first_batch_indexs,first_batch_values,second_batch_indexs,second_batch_values): # 找出index 
        offset_mapping  = batch_object_offset_mapping[i] # 拿到当前的offset_mapping
        origin_info = batch_object_origin_info[i]
        origin_text = origin_info['text']
        tokens = tokenizer.convert_ids_to_tokens(batch_object_input_ids[i]) # 得到原字符串
        objects = [] # 预测出最后的结果
        labels = []
        cur_object = ""
        detail_info = [(fi.item(),round(fv.item(),3),si.item(),round(sv.item(),3),t) for fi,fv,si,sv,t in zip(first_indexs,first_values,second_indexs,second_values,tokens)]
        for j,ind in enumerate(first_indexs):
            if ind > 1 : # 说明是一个标签的开始
                offset = offset_mapping[j]
                left,right = tuple(offset)
                if(cur_object!=""): # 说明紧接着
                    if ( (len(cur_object)> 1)
                     or (len(cur_object)==1 and (cur_object in country or cur_object in country))
                    ):
                        objects.append(cur_object)
                        cur_object_label = cur_object_label.replace("#","")                    
                        labels.append(cur_object_label)
                    cur_object = ""

                # 判断上一个字符是否是字母结束（英文）/数字并且当前的是否不是#开头 => 需要安排一个空格
                # if (( is_english_char(cur_object[-1]) or ('9'>= cur_object[-1] and '0'<= cur_object[-1]))
                #     and not(tokens[j].startswith("#"))
                #     and is_english_char(origin_text[left]) # 如果其后也是英文 
                #     and origin_text[left] != '-' # 不是连字符
                #     ):
                #     cur_object+=" "
                cur_object += origin_text[left:right]
                #cur_object += origin_text[i-1] # 因为有的字无法识别，所以这里用origin_text. i-1 是因为 相比而言，origin_text 少了 [CLS]
                cur_object_label = id2object_map[str(ind.item())]

            if ind == 1 and cur_object!="": # 说明是中间部分，且 cur_object 不为空
                offset = offset_mapping[j]
                left,right = tuple(offset)
                if( (is_english_char(cur_object[-1]) or ('9'>= cur_object[-1] and '0'<= cur_object[-1]))
                    and not(tokens[j].startswith("#"))
                    and is_english_char(origin_text[left]) # 如果其后也是英文
                    and origin_text[left] !='-'
                    ):
                    cur_object+=" "
                cur_object += origin_text[left:right]
            elif ind == 0 and cur_object!="": # 将 cur_object 放入到 objects 中
                cur_object = cur_object.replace("#","") # 替换掉，因为这会干扰后面的实现
                cur_object = cur_object.strip("《》，。+-.:：（）()、/\\！!") # 剃掉两边的所有符号
                # 后处理部分之删除不符合规则的数据
                # 后处理部分之判断object 的长度：如果长度大于1的才放进去
                if ( (len(cur_object)> 1)
                     or (len(cur_object)==1 and (cur_object in country or cur_object in country))
                    ):
                    objects.append(cur_object)
                    cur_object_label = cur_object_label.replace("#","")                    
                    labels.append(cur_object_label)
                cur_object = ""
            

        # 之前没有找到，那么就采取较为宽容的策略
        if len(objects) == 0:
            up_threshold = 6 # 间外
            down_threshold = 4 # 间内
            val_threshold = 3 # 单个值也必须满足一定的要求
        else:
            up_threshold = 6 # 间外
            down_threshold = 4 # 间内
            val_threshold = 3
        # 从次大的下标中寻找，此时需要注意其得分情况
        cur_object = ""
        k = 0
        pre_distance = 0 # 表示的是上一次fi_val 和 se_val 之间的距离差值
        cur_distance = 0 # 当前两者的差值
        for fi_ind,fi_val,se_ind,se_val in zip(first_indexs,first_values,second_indexs,second_values):
            cur_distance = abs(se_val-fi_val)

            # 第一种结束方式——说明是分界点
            if (abs(cur_distance-pre_distance) > up_threshold 
                and k # 排除掉CLS向量
                ): 
                if cur_object!="":
                    cur_object = cur_object.replace("#","") # 替换掉，因为这会干扰后面的实现
                    flag = 1
                    cur_object = cur_object.strip("《》，。+-.:：（）()、/\\！!") # 剃掉两边的所有符号
                    for have in objects: # 判断当前此轮预测的结果是否出现在之前的预测结果中
                        if (have.startswith(cur_object)
                            or have.find(cur_object)!=-1 # 如果该串作为子串出现过
                            ):
                            flag = 0
                            break
                    # 后处理部分之删除不符合规则的数据
                    # 后处理部分之判断object 的长度：如果长度大于1的才放进去
                    if (len(cur_object) > 1
                        and flag
                        ):                        
                        objects.append(cur_object)
                        cur_object_label = "top_2"
                        labels.append(cur_object_label)
                    cur_object = ""
                else: # 说明是一个标签的开始
                    if (fi_ind<=1 # 如果它在top_1 中就已经是标签了，则要过滤
                        and cur_distance < pre_distance
                        and se_val > 3
                        ): 
                        offset = offset_mapping[k]
                        left,right = tuple(offset)
                        # 判断上一个字符是否是字母结束（英文）/数字并且当前的是否不是#开头 => 需要安排一个空格
                        if (cur_object!="" 
                            and ( is_english_char(cur_object[-1]) or ('9'>= cur_object[-1] and '0'<= cur_object[-1]))
                            and not(tokens[k].startswith("#"))
                            and is_english_char(origin_text[left]) # 如果其后也是英文 
                            and origin_text[left] != '-' # 不是连字符
                            ):
                            cur_object+=" "
                        cur_object += origin_text[left:right]            
                        cur_object_label = id2object_map[str(se_ind.item())]
                    else:
                        cur_object = ""
            
            # 说明是中间部分
            elif (cur_object!="" 
                and abs(cur_distance - pre_distance) < down_threshold                
                ): 
                if( se_val > val_threshold# 当前也要过关
                    and fi_ind <= 1
                    ): 
                    offset = offset_mapping[k]
                    left,right = tuple(offset)
                    if( 
                        (is_english_char(cur_object[-1]) or ('9'>= cur_object[-1] and '0'<= cur_object[-1]))
                        and not(tokens[k].startswith("#"))
                        and is_english_char(origin_text[left]) # 如果其后也是英文
                        and origin_text[left] != '-'
                        ):
                        cur_object+=" "
                    cur_object += origin_text[left:right]
                else:
                    cur_object = ""
            
            # 第二种结束标志
            # 结束的标志要严格， 所以这里用or，两者
            elif(abs(cur_distance-pre_distance) > down_threshold  # 超过间内的距离
                or se_val < val_threshold
                ): 
                if cur_object!="":
                    cur_object = cur_object.replace("#","") # 替换掉，因为这会干扰后面的实现
                    flag = 1
                    cur_object = cur_object.strip("《》，。+-.:：（）()、/\\！!") # 剃掉两边的所有符号                        
                    for have in objects: # 判断当前此轮预测的结果是否出现在之前的预测结果中
                        if (have.startswith(cur_object)
                            or have.find(cur_object)!=-1 # 如果该串作为子串出现过
                            ):
                            flag = 0
                            break
                    # 后处理部分之删除不符合规则的数据
                    # 后处理部分之判断object 的长度：如果长度大于1的才放进去
                    if ( (len(cur_object) > 1 or (len(cur_object)==1 and (cur_object in country or cur_object in number)))
                        and flag
                        ):                      
                        objects.append(cur_object)
                        cur_object_label = "top_2"
                        labels.append(cur_object_label)
                    cur_object = ""
            k+=1
            pre_distance = cur_distance

        batch_objects.append(objects)
        batch_labels.append(labels)
        i+=1 

    return batch_objects,batch_labels




"""
功能： 将最后的结果解码并输出，这个函数是在 relation2id.json 中适用的，同样适用于 predicate2id.json 文件
"""
def decode_relation_class(logits,id2relation_map):
    m = LogSoftmax(dim=-1)
    a = m(logits)
    indexs = a.argmax(-1) # 找出每个batch的属性下标
    clas = []
    # for idx in indexs:
    #     cur_cls = id2relation_map[str(idx.item())]
    #     clas.append(cur_cls)      

    # 添加top_2 策略
    #logits # size [batch_size, relation_class_num]
    threshold = 11
    temp = t.topk(logits,k=2,dim=-1)
    value,index = temp
    index = index.transpose(1,0)
    second_indexs = index[1] # 排第二的index
    value = value.transpose(1,0)
    second_values = value[1]

    for item in zip(indexs,second_indexs,second_values):
        first_idx,second_idx,second_value = item
        if first_idx.item() == 0 and  second_value > threshold: # 代表是O类            
                cur_cls = id2relation_map[str(second_idx.item())]
                clas.append(cur_cls)        
        else:
            cur_cls = id2relation_map[str(first_idx.item())]
            clas.append(cur_cls)
    return clas

    
"""
功能： 将最后的结果组装成一个 spo_list，支持batch操作
"""
def post_process(batch_subjects,
                     batch_subjects_labels,
                     batch_objects,
                     batch_objects_labels,
                     batch_relations,
                     batch_origin_info
                     ):
    batch_res = []    
    cnt = 0
    cur_index = 0
    # step1.从所有的 batch 中取出一条样本
    for item_1 in zip(batch_subjects,batch_subjects_labels,batch_origin_info): 
        subjects,subject_labels,origin_info = item_1        
        cur_res ={} # 重置
        cur_res['text'] = origin_info['text']
        spo_list = [] # 是一个列表
        if len(subjects)==0: # 如果subjects 的结果为空
            # cnt += len(batch_objects[cur_index]) # 因为subjects 为空时，是没有训练数据的
            cur_index += 1
            continue
        # step2. 从某条样本中取出所有的 subjects 以及其标签
        for item_2 in zip(subjects,subject_labels):  
            subject,subject_label = item_2
            # 取该subjects 对应的objects 和 objects_labels
            objects = batch_objects[cur_index]  # 和subject 对应在一起的所有 object
            objects_labels = batch_objects_labels[cur_index]
            
            # step3. 从上述的 objects 以及labels 中对应取出单个
            for item_3 in zip(objects,objects_labels): 
                cur_dict = {} # cur_dict 都会被放入到spo_list 中
                obj , obj_label = item_3
                cur_dict['predicate'] = batch_relations[cnt]
                
                # 说明预测的subject 和 object 一样，这样的数据没有意义
                # 或者预测结果表明二者没有关系
                if (subject == obj or cur_dict['predicate']=='O'):
                    cnt+=1
                    continue
                val_1 = {} # 存放object
                val_2 = {} # 存放object_type
                
                
                # 如果是复杂的结构，则需要另行处理
                if(batch_relations[cnt] == '饰演' or batch_relations[cnt]=="配音"): 
                    if obj_label == "影视作品":
                        val_1["inWork"] = obj
                        val_2["inWork"] = obj_label
                    elif obj_label == "人物":
                        val_2["@value"] = obj_label
                        val_1["@value"] = obj
                else: 
                    val_1["@value"] = obj
                    val_2["@value"] = obj_label
                 
                cur_dict['object'] = val_1
                cur_dict['object_type'] = val_2
                cur_dict['subject_type'] = subject_label
                cur_dict['subject'] = subject                    
                #print(cur_res)
                cnt += 1
                spo_list.append(cur_dict)                
            cur_index += 1
            
            # 处理复杂的结构 => 修改 spo_list            
            temp = []
            visit = [1]* len(spo_list) # 用于标记是否copy值到其中
            for i,spo_i in enumerate(spo_list):        
                for j in range(i+1,len(spo_list)):
                    #print(spo_i,spo_list[j],"\n")
                    spo_j = spo_list[j]
                    # ======================== 如果是饰演关系   ========================
                    if spo_i['subject'] == spo_j['subject'] and spo_i['predicate'] == spo_j['predicate'] and spo_i['predicate'] == "饰演":# 说明可以合并，合并前记得删除
                        visit[i] = 0
                        visit[j] = 0 #用于标记是否copy值
                        combine_res = {}
                        combine_res['subject'] = spo_i['subject']
                        combine_res['subject_type'] = spo_i['subject_type']
                        object_1 = spo_i['object'] # dict
                        object_2 = spo_j['object']
                        object_2.update(object_1)            
                        object_type_1 = spo_i['object_type'] # dict
                        object_type_2 = spo_j['object_type']
                        object_type_2.update(object_type_1)        
                        combine_res['object_type'] = object_type_2
                        combine_res['object'] = object_2        
                        combine_res['predicate'] = spo_i['predicate']
                        #print(combine_res,"\n")
                        temp.append(combine_res)
                    elif spo_i['subject'] == spo_j['subject'] and spo_i['predicate'] == spo_j['predicate'] and spo_i['predicate'] == "配音":
                        visit[i] = 0
                        visit[j] = 0 #用于标记是否copy值
                        combine_res = {}
                        combine_res['subject'] = spo_i['subject']
                        combine_res['subject_type'] = spo_i['subject_type']
                        object_1 = spo_i['object'] # dict
                        object_2 = spo_j['object']
                        object_2.update(object_1)
                        object_type_1 = spo_i['object_type'] # dict
                        object_type_2 = spo_j['object_type']
                        object_type_2.update(object_type_1)        
                        combine_res['object_type'] = object_type_2
                        combine_res['object'] = object_2
                        combine_res['predicate'] = spo_i['predicate']
                        #print(combine_res,"\n")
                        temp.append(combine_res)
            for i in range(0,len(visit)):
                if visit[i]:
                    temp.append(spo_list[i])            
            cur_res["spo_list"]= temp
            
        batch_res.append(cur_res)
    return batch_res
    

"""
第2版后处理函数
01.支持batch操作
"""
def post_process_2(batch_subjects,                     
                     batch_objects,                     
                     batch_relations,
                     batch_origin_info
                    ):
    batch_res = []
    cnt = 0
    cur_index = 0
    # step1.从所有的 batch 中取出一条样本
    for item_1 in zip(batch_subjects,batch_origin_info): 
        subjects,origin_info = item_1        
        cur_res ={} # 重置
        cur_res['text'] = origin_info['text']
        spo_list = [] # 是一个列表
        if len(subjects)==0: # 如果subjects 的结果为空
            
            cur_index += 1
            continue
        # step2. 从某条样本中取出所有的 subjects 以及其标签
        for subject in subjects:
            # 取该subjects 对应的objects 和 objects_labels
            objects = batch_objects[cur_index]  # 和subject 对应在一起的所有 object                        
            # step3. 从上述的 objects 以及labels 中对应取出单个
            for obj in objects: 
                cur_dict = {} # cur_dict 都会被放入到spo_list 中                
                cur_dict['predicate'] = batch_relations[cnt]
                
                # 说明预测的subject 和 object 一样，这样的数据没有意义
                # 或者预测结果表明二者没有关系
                if (subject == obj or cur_dict['predicate']=='O'):
                    cnt+=1
                    continue
                val_1 = {} # 存放object                
                if cur_dict['predicate'] in ['上映时间_@value','上映时间_inArea','饰演_@value','饰演_inWork','获奖_period','获奖_@value','获奖_inWork','获奖_onDate','配音_@value','配音_inWork','票房_@value','票房_inArea']:                                        
                    predicate = cur_dict['predicate']
                    left,right = tuple(predicate.split("_")) # 拿到左右两个
                    val_1[right] = obj
                    cur_dict['predicate'] = left
                else:
                    val_1['@value'] = obj
                 
                cur_dict['object'] = val_1                
                cur_dict['subject'] = subject                    
                #print(cur_res)
                cnt += 1
                spo_list.append(cur_dict)                
            cur_index += 1
        
        # 合并复杂结构
        combine_res = {} # 最后的结果
        for spo in spo_list:
            predicate = spo['predicate']
            subject = spo['subject']
            obj_key = spo['object'].keys()
            obj_val = spo['object'].values()
            if subject+"_"+predicate not in combine_res.keys():        
                combine_res[subject+"_"+predicate]=[]
                combine_res[subject+"_"+predicate].append(spo['object'])
            else:
                combine_res[subject+"_"+predicate].append(spo['object'])        

        
        temp = []
        for item in combine_res.items():
            key ,value = item            
            line = key.split("_")
            predicate = line[-1]
            subject = "_".join(line[0:-1])
            if predicate in ['上映时间','饰演','获奖','配音','票房']: # 
                cur_temp = {}
                cur_temp['object'] = {} # 空字典
                cur_temp['subject'] = subject
                cur_temp['predicate'] = predicate
                for val in value:
                    cur_temp['object'].update(val)
                    #{'onDate': '2001年'}, {'@value': '中国原创音乐榜千禧全国成就大奖'}
                temp.append(cur_temp)
            else:                
                for val in value:
                    cur_temp = {}
                    cur_temp['subject'] = subject
                    cur_temp['predicate'] = predicate
                    cur_temp['object'] = {}
                    cur_temp['object'].update(val)
                    temp.append(cur_temp)            
        cur_res["spo_list"]= temp
        batch_res.append(cur_res)
    return batch_res





"""
去除字符串中的数字，如果有连续的数字仅保留一个。这个函数作废，没用到。
"""
def get_rid_of_number_in_str(string):
    deleted = [1] * len(string)
    res = "" 
    pre_char = '' # 上一个字符
    for i,char in enumerate(string):
        if char.isdigit()  and pre_char.isdigit():# 如果当前是数字，且之前也是数字
            deleted[i] = 0 # 记为待删除
        #print(char,end='')
        pre_char = char
    
    for i in range(len(string)):
        if deleted[i]:
            res+=string[i]
    return res


"""后处理部分
功能：添加书名号中的内容

特殊样例：《库洛洛版《一个陌生女人的来信》》
"""
def addBookName(text):
    target = [] 
    cur_text = ""
    flag = 0 # 是否在书名号中
    pre_flag = 1
    for char in text:
        if flag == 1 and char!='\n' :
            if char!="》":
                cur_text+=char
            if char == "》" and not pre_flag:
                cur_text +=char

        if char == "《" and flag == 0: # 第一次碰到 《
            flag = 1        
        elif char =="》" and pre_flag and flag:
            cur_text = cur_text.strip("\n") # 去掉换行
            cur_text = cur_text.strip(" ") # 去掉空格
            target.append(cur_text)
            cur_text = ""
            flag = 0
        elif flag == 1 and char == "《":
            pre_flag = 0 # 说明中间还有一个 《
        elif flag ==1 and char == "》" and pre_flag == 0:
            pre_flag = 1
    target = list(set(target))
    return target


"""后处理部分
功能：删除subject 为 XXXX 年 XX 月 XX 日这种格式的数据
"""
def is_year_month_day(subject):
    if ('年' in subject and '月' in subject and '日' in subject):
        return True
    if ('年' in subject and '月' in subject ):
        return True
    if ('月' in subject and '日' in subject):
        return True
    return False


"""
可视化subject的预测
"""
def visualize_subject(file_path,all_subjects):
    with open(file_path,'w') as f:
        for batch_subjects in all_subjects:
            for subjects in batch_subjects:                                
                if len(subjects) == 0:
                    f.write("None"+"\n")
                else:
                    for subject in subjects:
                        f.write(subject +"\n")                    
                f.write("\n")



"""
可视化subject的预测
"""
def visualize_subject_with_label(file_path,all_subjects,all_subject_labels):
    with open(file_path,'w') as f:
        for item_1 in zip(all_subjects,all_subject_labels): # 取出一个batch
            batch_subjects,batch_subject_labels = item_1
            for item in (zip(batch_subjects,batch_subject_labels)): # 从batch中取出一个sample
                subjects,labels = item                
                if len(subjects) == 0:
                    f.write("None"+"\n")
                else:
                    for lines in zip(subjects,labels): # 依次从每个sample 取出所有subject
                        subject,label = lines
                        f.write(subject +"\t" +label+"\n")                
                
                # 下面这行代码无论是if,else 都得执行
                f.write("\n")

"""
可视化subject 和 object 的预测结果
01.追加写入
"""
def visualize_subject_object(file_path,batch_subjects,batch_objects):    
    all_subjects = []
    for subjects in batch_subjects:
        if len(subjects) == 0:
            all_subjects.append([]) # 放一个空的list进去
        else:
            all_subjects.extend(subjects)

    with open(file_path,'a') as f:
        index = 0
        for subjects in (batch_subjects):# 找出每条句子的所有subject
            if len(subjects) == 0:
                subjects = ["None"] # 手动添加一个None                
            for subject in subjects: # 找出该句子的某个subject 
                cur_objects = batch_objects[index]
                for object in cur_objects:
                    if object == subject:
                        continue
                    line = subject + "\t" + object +"\n"
                    f.write(line)
                index += 1
            # 在每条句子后需要一个换行
            f.write("\n") 


"""
这个方法和上面这个因为batch_subjects 中的内容不同
"""
def visualize_subject_object_2(file_path,batch_subjects,batch_objects):        
    with open(file_path,'a') as f:
        for item in zip(batch_subjects,batch_objects):# 找出每条句子的所有subject
            subject,objects = item
            if len(subject) == 0:
                subject = "None" # 手动添加一个None
            if len(objects) == 0:
                line = subject + "\t" + "None" +"\n"
                f.write(line)
            for object in objects:
                if object == subject:
                    continue
                line = subject + "\t" + object +"\n"
                f.write(line)                
        
        # 在每条样本后需要一个换行
        f.write("\n")




"""
仅仅写入object，而不写入subject
"""
def visualize_object(file_path,batch_objects):        
    with open(file_path,'a') as f:
        for objects in batch_objects:# 找出每条句子的所有subject                 
            if len(objects) == 0:
                f.write("None")
            for object in objects:
                line = object +"\t"
                f.write(line)        
        # 在每条样本后需要一个换行
            f.write("\n\n")


"""
可视化object的预测结果
"""
def visualize_object_2(pred_file_path, all_objects, all_object_labels):
    with open(pred_file_path,'w') as f:
        for item in zip(all_objects,all_object_labels):
            objects,labels = item
            f.write(str(objects) +"\n")
            f.write(str(labels) + "\n")
            f.write("\n")



"""
功能：从train_data.json中得到所有的subject集合
01.过滤subject 长度=1 的
"""
def get_all_subjects(train_data_path):
    all_subject = set()
    with open(train_data_path) as f:
        line = f.readline()
        while(line):
            line = json.loads(line) # 加载成json
            spo_list = line['spo_list']
            for spo in spo_list:
                subject = spo['subject']
                if len(subject) > 1: # 过滤掉一个字的
                    all_subject.add(subject)           
            line = f.readline()
    #print(len(all_subject))    
    return all_subject


"""
将两个预测文件的预测值合并在一起
"""
def combine_all_pred(pred_file_1,pred_file_2):
    cont = []
    out_file_path = './data/fin_res.json'
    res1_map ={} # text => pred
    res2_map = {}
    # 逐行合并
    with open(pred_file_1,'r') as f1:        
        line1 = f1.readline()        
        while(line1):
            line1 = json.loads(line1)            
            text = line1['text']
            spo_list_1 = line1['spo_list'] # list
            res1_map[text] = spo_list_1
            line1 = f1.readline()            
    
    with open(pred_file_2,'r') as f2:
        line2 = f2.readline()
        while(line2):        
            line2 = json.loads(line2)
            text = line2['text']
            spo_list_2 = line2['spo_list']
            if text in res1_map.keys():
                spo_list_2.extend(res1_map[text])
            cont.append({"text":text,"spo_list":spo_list_2})
            line2 = f2.readline()



    with open(out_file_path,'w') as f:
        for line in cont:
            json.dump(line,f,ensure_ascii=False)
            f.write("\n")


# 添加所有的国家 object
def get_all_country(train_data_path):
    all_country = set()
    with open(train_data_path) as f:
        line = f.readline()
        while(line):
            line = json.loads(line) # 加载成json
            spo_list = line['spo_list']
            for spo in spo_list:
                country = spo['object']
                predicate = spo['predicate'] #
                if predicate == '国籍' :
                    all_country.add(country['@value'])
            line = f.readline()
    for country in all_country:
        if len(country) == 1:
            print(country)
    return all_country


# 对国籍关系的后处理
def add_relation_of_country(batch_subjects,
                batch_subject_labels,
                batch_objects,
                batch_object_labels,
                batch_relations,
                batch_origin_info
                ):    
    cnt = 0
    cur_index = 0
    # step1.从所有的 batch 中取出一条样本
    for item_1 in zip(batch_subjects,batch_subject_labels,batch_origin_info): 
        subjects,subject_labels,origin_info = item_1
        cur_res ={} # 重置
        cur_res['text'] = origin_info['text']
        
        if len(subjects)==0: # 如果subjects 的结果为空
            cur_index += 1
            continue
        
        # step2. 从某条样本中取出所有的 subjects 以及其标签
        for item2 in zip(subjects,subject_labels):
            subject,subject_label = item2
            # 取该subjects 对应的objects 和 objects_labels
            objects = batch_objects[cur_index]  # 和subject 对应在一起的所有 object
            object_labels = batch_object_labels[cur_index] # 取labels
            # step3. 从上述的 objects 以及labels 中对应取出单个
            for item3 in zip(objects,object_labels): 
                obj,obj_labels = item3
                cur_dict = {} # cur_dict 都会被放入到spo_list 中                
                cur_dict['predicate'] = batch_relations[cnt]
                
                # 说明预测的subject 和 object 一样，这样的数据没有意义
                # 或者预测结果表明二者没有关系
                if cur_dict['predicate']=='O':
                    if subject_label in  ['人物','娱乐人物'] and obj_labels == '国家':
                        batch_relations[cnt] = '国籍'                        
                cnt += 1                
            cur_index += 1                                
    return batch_relations


# 根据作词作曲添加relation，这是一个有效的后处理
def add_write_ci_relation(pred_file_path):
    out_path = "/home/lawson/program/DuIE_py/data/test_data_predict_001.json_valid_2.json"
    all = []
    cnt = 0
    with open(pred_file_path,'r') as f:
        line = f.readline()        
        while(line):
            line = json.loads(line)
            text = line['text']            
            spo_list = line['spo_list']
            temp_spo_list = copy.copy(spo_list)
            if "作词作曲" in text: # 连着出现            
                cur_spo = {}
                for spo in spo_list:
                    subject = spo['subject']
                    predicate = spo['predicate']
                    if predicate == "作词": # 那么添加一个作曲到其中
                        cur_spo['subject'] = subject
                        cur_spo['predicate'] = '作曲'
                        cur_spo['object'] = spo['object']
                        temp_spo_list.append(cur_spo)
                        cnt+=1
            cur = {}
            cur['text'] = text
            cur['spo_list'] = temp_spo_list
            all.append(cur)
            line = f.readline()
    print(f"添加作曲关系{cnt}")
    with open(out_path,'w') as f:
        for line in all:
            json.dump(line,f,ensure_ascii=False)
            f.write("\n")



"""
将文本中的空格全部替换成句号
01.书名号中的空格不能替换
02.如果是符号后的空格，则直接压缩。
"""
def replace_space2period(in_data_path,out_data_path):
    punctuation = [ "。","？", "！", "，","、", "；", "：","‘",
             "’", "“", "”", "（","）", "〔", "〕", "【", "】", "—", "…","–", "―", '《', '》', '．', # chinese
             ',','.',':', '-'# english
             ] 
    invalid = []
    total = 0
    after = []
    with open(in_data_path,'r') as f:
        line = f.readline()
        while(line):
            line = json.loads(line)
            text = line['text']
            #spo_list = line['spo_list']
            #不能直接采用公式去压缩空格
            text = re.sub('\s+',' ',text) # 将多个空格转换成一个空格
            cur_text = ""
            in_bookname = 0 # 前面《 号的个数
            pre_is_punctuation = False # 上一个字符是否是标点符号
            cnt = 0 # 为手动添加的句号计数
            pre_word = '' # 上一个字符
            for i in range(len(text)):
                word = text[i]
                if i != len(text) -1 : # 预判后一个符号
                    next_word = text[i+1] 
                else:
                    next_word = "end"
                if word == " " or word =='\xa0': #是空格
                    if in_bookname: #且在书名号中
                        cur_text += word
                    elif ((not pre_is_punctuation)# 上一个词不是标点符号
                            and (next_word not in punctuation) # 接下来的一个词也不是标点符号
                            and not ( is_english_char(pre_word) or ('9'>= pre_word and '0'<= pre_word)) # 上一个字符不是英文字符或数字
                        ):
                        cur_text += '，' # 替换成逗号
                        cnt += 1
                        pre_is_punctuation = True # 保证多个空格只被替换一次
                    elif (
                        (is_english_char(pre_word) or ('9'>= pre_word and '0'<= pre_word)) # 当前是英文字符或数字
                        and ( is_english_char(next_word) or ('9'>= next_word and '0'<= next_word)) # 下一个字符不是英文字符或数字
                    ):
                        cur_text +=' ' #加上空格
                        
                elif word !=" " :
                    cur_text += word
                    if word == "《":
                        in_bookname += 1
                    elif word == "》":
                        in_bookname -= 1 

                    # 单独判断本word是否是标点符号
                    if word in punctuation:
                        pre_is_punctuation = True
                    else:
                        pre_is_punctuation = False
                                
                pre_word = word
            cur = {}
            if cnt < 15 and cnt:
                cur['text'] = cur_text
                total += 1
                after.append(cur_text)
                invalid.append(cur)
            else:
                cur['text'] = text
            #cur['spo_list'] = spo_list
            
            line = f.readline()
    print(f"改动的文本有：{total}个")
    # with open(out_data_path,'w') as f:
    #     for line in valid:
    #         json.dump(line,f,ensure_ascii=False)
    #         f.write("\n")

    with open('./test_data_modify.text','w') as f:
        for line in invalid:
            json.dump(line,f,ensure_ascii=False)
            f.write("\n")
            
"""
转换文本，将文本中的多余空格替换成有规律的，
"""
def transpose_text(text):    
    punctuation = [ "。","？", "！", "，","、", "；", "：","‘",
             "’", "“", "”", "（","）", "〔", "〕", "【", "】", "—", "…","–", "―", '《', '》', '．', # chinese
             ',','.',':', '-'# english
             ] 
    
    
    #不能直接采用公式去压缩空格
    text = re.sub('\s+',' ',text) # 将多个空格转换成一个空格
    cur_text = ""
    in_bookname = 0 # 前面《 号的个数
    pre_is_punctuation = False # 上一个字符是否是标点符号
    cnt = 0 # 为手动添加的句号计数
    pre_word = '' # 上一个字符
    for i in range(len(text)):
        word = text[i]
        if i != len(text) -1 : # 预判后一个符号
            next_word = text[i+1] 
        else:
            next_word = "end"
        if word == " " or word =='\xa0': #是空格
            if in_bookname: #且在书名号中
                cur_text += word
            elif ((not pre_is_punctuation)# 上一个词不是标点符号
                    and (next_word not in punctuation) # 接下来的一个词也不是标点符号
                    and not ( is_english_char(pre_word) or ('9'>= pre_word and '0'<= pre_word)) # 上一个字符不是英文字符或数字
                ):
                cur_text += '，' # 替换成逗号
                cnt += 1
                pre_is_punctuation = True # 保证多个空格只被替换一次
            elif (
                (is_english_char(pre_word) or ('9'>= pre_word and '0'<= pre_word)) # 当前是英文字符或数字
                and ( is_english_char(next_word) or ('9'>= next_word and '0'<= next_word)) # 下一个字符不是英文字符或数字
            ):
                cur_text +=' ' #加上空格
                
        elif word !=" " :
            cur_text += word
            if word == "《":
                in_bookname += 1
            elif word == "》":
                in_bookname -= 1 

            # 单独判断本word是否是标点符号
            if word in punctuation:
                pre_is_punctuation = True
            else:
                pre_is_punctuation = False                            
        pre_word = word        
    if cnt < 15 and cnt:
        return cur_text    
    else:
        return "1111"
    



"""
more_path:表示的是文件较大的那个
less_path:表示的是文件较小的那个（也就是经过错误数据生成得到的）
"""
def get_correct_result(more_path,less_path):
    less_res = {} # text => spo的一个dict

    with open(less_path, 'r') as f:
        line = f.readline()
        while(line):
            line = json.loads(line)
            text = line['text']
            spo_list = line['spo_list']
            less_res[text] = spo_list
            line = f.readline()
    
    
    fin_res = []    
    with open(more_path, 'r') as f:
        line = f.readline()        
        while(line):
            line = json.loads(line)
            text = line['text']            
            # 执行替换操作
            a = transpose_text(text)
            if a in less_res.keys():          
                line['spo_list'].extend(less_res[a])                
            fin_res.append(line)
            line = f.readline()
    
    # 写入最后的结果
    with open('./fin_res.json','w') as f:
        for line in fin_res:
            json.dump(line,f,ensure_ascii=False)
            f.write("\n")
    
#5000 个
def analysis_text(file_path):
    cnt = 0
    with open(file_path,'r') as f:
        line = f.readline()
        while(line):
            line = line.strip("\n")
            #print(line)
            if len(line) == 1:
                cnt+=1
            line = f.readline()    
    print(cnt)


if __name__ == "__main__":
    #get_precision_recall_f1(golden_file="./data/dev_data.json",
     #                       predict_file="./data/predictions.json.zip")
    text = "库洛洛版《一个陌生女人的来信》,《步步惊心》是lotus丝莲创作的网络小说，发表于晋江文学网"
    #target = addBookName(text)
    #print(target)
    # get_all_subjects(
    #     all_subject_path=None,
    #     train_data_path="./data/train_data.json"
    # )
    get_all_country(train_data_path="./data/train_data.json")
    #add_write_ci_relation(pred_file_path="./data/test_data_predict_001.json_valid.json")
    # in_data_path='./data/test_data.json'
    # out_data_path='./data/test_data_valid.json'
    # if os.path.exists(out_data_path):
    #     os.remove(out_data_path)
    #replace_space2period(in_data_path,out_data_path)
    # get_correct_result(more_path="/home/lawson/program/DuIE_py/data/test_data_predict_556673_9.json_valid.json",
    # less_path="/home/lawson/program/DuIE_py/data/test_data_modify_predict_556673_9.json")
    #combine_all_pred(pred_file_1='./data/test_data_predict_valid_71.71.json',pred_file_2='./data/test_data_predict_556673_20210428_11.json_valid.json')
    #analysis_text("./data/dev_data_5000_object_predict.txt_forget.text")