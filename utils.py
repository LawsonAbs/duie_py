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

import codecs
import json
import os
import re
import zipfile
import torch as t
from torch.nn import LogSoftmax
import numpy as np

"""
是否是英文字符或者是连字符
"""
def is_english_char_or_(ch):
    if ('z' >= ch and 'a' <= ch ):
        return True
    if ('Z' >= ch and 'A' <= ch):
        return True
    if (ch == '-'):
        return True
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
适用的情况是 不同的B标签，统一的I/O标签，
params:
 origin_text:原文本，用于消除不认识的字变成 [UNK] 的问题

"""
def decode_subject(logits,id2subject_map,input_ids,tokenizer,batch_origin_info,batch_offset_mapping):
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
                    and ( is_english_char_or_(cur_subject[-1]) or ('9'>= cur_subject[-1] and '0'<= cur_subject[-1]))
                    and not(tokens[j].startswith("#"))
                    and is_english_char_or_(origin_text[left]) # 如果其后也是英文 
                    ):
                    cur_subject+=" "
                cur_subject += origin_text[left:right]
                #cur_subject += origin_text[i-1] # 因为有的字无法识别，所以这里用origin_text. i-1 是因为 相比而言，origin_text 少了 [CLS]
                cur_subject_label = id2subject_map[str(ind.item())]
            if ind == 1 and cur_subject!="": # 说明是中间部分，且 cur_subject 不为空
                offset = offset_mapping[j]
                left,right = tuple(offset)
                if( 
                    (is_english_char_or_(cur_subject[-1]) or ('9'>= cur_subject[-1] and '0'<= cur_subject[-1]))
                    and not(tokens[j].startswith("#"))
                    and is_english_char_or_(origin_text[left]) # 如果其后也是英文
                    ):
                    cur_subject+=" "
                cur_subject += origin_text[left:right]
            elif ind == 0 and cur_subject!="": # 将 cur_subject 放入到 subjects 中
                cur_subject = cur_subject.replace("#","") # 替换掉，因为这会干扰后面的实现
                # 后处理部分之删除不符合规则的数据
                # 后处理部分之判断subject 的长度：如果长度大于1的才放进去
                if (not is_year_month_day(cur_subject) 
                    #and (len(cur_subject)> 1)
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
                    (is_english_char_or_(cur_subject[-1]) or ('9'>= cur_subject[-1] and '0'<= cur_subject[-1]))
                    and not(tokens[j].startswith("#"))
                    and is_english_char_or_(origin_text[left]) # 如果其后也是英文
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
                        and ( is_english_char_or_(cur_subject[-1]) or ('9'>= cur_subject[-1] and '0'<= cur_subject[-1]))
                        and not(tokens[j].startswith("#"))
                        and is_english_char_or_(origin_text[left]) # 如果其后也是英文 
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
                        and ( is_english_char_or_(cur_subject[-1]) or ('9'>= cur_subject[-1] and '0'<= cur_subject[-1]))
                        and not(tokens[j].startswith("#"))
                        and is_english_char_or_(origin_text[left]) # 如果其后也是英文 
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
                        and ( is_english_char_or_(cur_subject[-1]) or ('9'>= cur_subject[-1] and '0'<= cur_subject[-1]))
                        and not(tokens[j].startswith("#"))
                        and is_english_char_or_(origin_text[left]) # 如果其后也是英文 
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
                        and ( is_english_char_or_(cur_subject[-1]) or ('9'>= cur_subject[-1] and '0'<= cur_subject[-1]))
                        and not(tokens[j].startswith("#"))
                        and is_english_char_or_(origin_text[left]) # 如果其后也是英文 
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
                        and ( is_english_char_or_(cur_subject[-1]) or ('9'>= cur_subject[-1] and '0'<= cur_subject[-1]))
                        and not(tokens[j].startswith("#"))
                        and is_english_char_or_(origin_text[left]) # 如果其后也是英文 
                        ):
                        cur_subject+=" "
                    cur_subject += origin_text[left:right]                    
                    
                elif cur_subject != "":
                    # 先把之前的内容存下来
                    offset = offset_mapping[j]
                    left,right = tuple(offset)
                    # 判断上一个字符是否是字母结束（英文）/数字并且当前的是否不是#开头 => 需要安排一个空格
                    if (cur_subject!="" 
                        and ( is_english_char_or_(cur_subject[-1]) or ('9'>= cur_subject[-1] and '0'<= cur_subject[-1]))
                        and not(tokens[j].startswith("#"))
                        and is_english_char_or_(origin_text[left]) # 如果其后也是英文 
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
                        and ( is_english_char_or_(cur_subject[-1]) or ('9'>= cur_subject[-1] and '0'<= cur_subject[-1]))
                        and not(tokens[j].startswith("#"))
                        and is_english_char_or_(origin_text[left]) # 如果其后也是英文 
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
def decode_object(logits,id2object_map,tokenizer,batch_object_input_ids,batch_object_origin_info,batch_object_offset_mapping):
    m = LogSoftmax(dim=-1)
    a = m(logits)
    batch_indexs = a.argmax(-1) # 在该维度找出值最大的，就是预测出来的标签
    batch_objects = [] # 预测出最后的结果
    batch_labels = []    
    for item in zip(batch_indexs,batch_object_input_ids,batch_object_offset_mapping,batch_object_origin_info):
        indexs,input_ids,offset , origin_info = item
        text_raw = origin_info['text']
        tokens = tokenizer.convert_ids_to_tokens(input_ids) # 得到原字符串
        objects = []
        labels = []
        cur_object = ""
        for i,ind in enumerate(indexs):
            if ind > 1 : # 说明是一个标签的开始                
                left,right = tuple(offset[i])
                # 判断上一个字符是否是字母结束（英文）/数字并且当前的是否不是#开头 => 需要安排一个空格
                if (cur_object!="" 
                    and ( is_english_char_or_(cur_object[-1]) or ('9'>= cur_object[-1] and '0'<= cur_object[-1]))
                    and not(tokens[i].startswith("#"))
                    and is_english_char_or_(text_raw[left]) # 如果其后也是英文 
                    ):
                    cur_object+=" "
                cur_object+= text_raw[left:right]
                cur_object_label = id2object_map[str(ind.item())]
            if ind == 1 and cur_object!="": # 说明是中间部分，且 cur_subject 不为空
                left,right = tuple(offset[i])
                if (cur_object!="" 
                    and ( is_english_char_or_(cur_object[-1]) or ('9'>= cur_object[-1] and '0'<= cur_object[-1]))
                    and not(tokens[i].startswith("#"))
                    and is_english_char_or_(text_raw[left]) # 如果其后也是英文 
                    ):
                    cur_object+=" "
                cur_object+= text_raw[left:right] 
            elif ind == 0 and cur_object!="": # 将 cur_subject 放入到 subjects 中
                cur_object = cur_object.replace("#","")                
                objects.append(cur_object)
                labels.append(cur_object_label)
                cur_object = ""
        batch_objects.append(objects)
        batch_labels.append(labels)
    # 然后再找出对应的内容
    return batch_objects,batch_labels




"""
功能： 将最后的结果解码并输出，这个函数是在 relation2id.json 中适用的，同样适用于 predicate2id.json 文件
"""
def decode_relation_class(logits,id2relation_map):
    m = LogSoftmax(dim=-1)
    a = m(logits)
    indexs = a.argmax(-1) # 找出每个batch的属性下标
    clas = []
    for idx in indexs:
        cur_cls = id2relation_map[str(idx.item())]
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
            cur_text = cur_text.strip("") # 去掉空格
            target.append(cur_text)
            cur_text = ""
            flag = 0
        elif flag == 1 and char == "《":
            pre_flag = 0 # 说明中间还有一个 《
        elif flag ==1 and char == "》" and pre_flag == 0:
            pre_flag = 1
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
                        if subject == "库洛洛版《一个陌生女人的来信":
                            print("xx")
                        f.write(subject +"\n")                    
                f.write("\n")



"""
可视化subject的预测
"""
def visualize_subject_label(file_path,all_subjects,all_subject_labels):
    with open(file_path,'w') as f:
        for item_1 in zip(all_subjects,all_subject_labels):
            batch_subjects,batch_subject_labels = item_1
            for item in (zip(batch_subjects,batch_subject_labels)):
                subjects,labels = item                
                if len(subjects) == 0:
                    f.write("None"+"\n")
                else:
                    for lines in zip(subjects,labels):
                        subject,label = lines
                        f.write(subject +"\t" +label+"\n")                
                
                # 下面这行代码无论是if,else 都得执行
                f.write("\n")

"""
可视化subject 和 object 的预测结果

"""
def visualize_subject_object(file_path,batch_subjects,batch_objects,batch_origin_info):    
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


if __name__ == "__main__":
    #get_precision_recall_f1(golden_file="./data/dev_data.json",
     #                       predict_file="./data/predictions.json.zip")
    text = "库洛洛版《一个陌生女人的来信》,《步步惊心》是lotus丝莲创作的网络小说，发表于晋江文学网"
    target = addBookName(text)
    print(target)