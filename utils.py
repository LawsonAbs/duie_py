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
"""
def decode_subject(logits,id2subject_map,input_ids,tokenizer):    
    m = LogSoftmax(dim=-1)
    a = m(logits)
    batch_indexs = a.argmax(2) # 在该维度找出值最大的，就是预测出来的标签    
    batch_subjects =[]
    batch_labels = []    
    for i,indexs in enumerate(batch_indexs):
        tokens = tokenizer.convert_ids_to_tokens(input_ids[i]) # 得到原字符串
        subjects = [] # 预测出最后的结果
        labels = []
        cur_subject = ""
        for i,ind in enumerate(indexs):
            if ind > 1 : # 说明是一个标签的开始
                cur_subject+=tokens[i] 
                cur_subject_label = id2subject_map[str(ind.item())]
            if ind == 1 and cur_subject!="": # 说明是中间部分，且 cur_subject 不为空
                cur_subject += tokens[i]
            elif ind == 0 and cur_subject!="": # 将 cur_subject 放入到 subjects 中
                cur_subject = cur_subject.replace("#","") # 替换掉，因为这会干扰后面的实现
                subjects.append(cur_subject)
                cur_subject_label = cur_subject_label.replace("#","")
                labels.append(cur_subject_label)
                cur_subject = ""
        batch_subjects.append(subjects)
        batch_labels.append(labels)
    # 然后再找出对应的内容
    return batch_subjects,batch_labels


"""
功能：由 预测object 的labels 得到object

params:
 logits: 预测的值，需要经过softmax处理，然后得到结果
 id2object_map
 ...

01.
"""
def decode_object(logits,id2object_map,tokenizer,object_input_ids):
    m = LogSoftmax(dim=-1)
    a = m(logits)
    batch_indexs = a.argmax(2) # 在该维度找出值最大的，就是预测出来的标签
    batch_objects = [] # 预测出最后的结果
    batch_labels = []    
    for item in zip(batch_indexs,object_input_ids):
        indexs,input_ids = item
        tokens = tokenizer.convert_ids_to_tokens(input_ids) # 得到原字符串
        objects = []
        labels = []
        cur_object = ""
        for i,ind in enumerate(indexs):
            if ind > 1 : # 说明是一个标签的开始
                cur_object+=tokens[i] 
                cur_object_label = id2object_map[str(ind.item())]
            if ind == 1 and cur_object!="": # 说明是中间部分，且 cur_subject 不为空
                cur_object += tokens[i]
            elif ind == 0 and cur_object!="": # 将 cur_subject 放入到 subjects 中
                cur_object = cur_object.replace("#","")
                cur_object_label = cur_object_label.replace("#","")

                objects.append(cur_object)                
                labels.append(cur_object_label)
                cur_object = ""
        batch_objects.append(objects)
        batch_labels.append(labels)
    # 然后再找出对应的内容
    return batch_objects,batch_labels

"""
功能： 将最后的结果解码并输出
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
功能： 将最后的结果组装成一个 spo_list，只支持单条预测!!!
"""
def post_process(batch_subjects,
                     batch_subjects_labels,
                     batch_objects,
                     batch_objects_labels,
                     batch_relations,
                     origin_info
                     ):
    batch_res = []    
    cnt = 0    
    for item_1 in zip(batch_subjects,batch_subjects_labels): # 从所有的 batch 中取出一条样本
        subjects,subject_labels = item_1
        cur_index = 0
        cur_res ={} # 重置
        cur_res['text'] = origin_info     
        spo_list = [] # 是一个列表   
        for item_2 in zip(subjects,subject_labels):  # 从某条样本中取出所有的 subjects 以及其标签
            subject,subject_label = item_2
            # 取该subjects 对应的objects 和 objects_labels
            objects = batch_objects[cur_index:cur_index+1][0]  # 和subject 对应在一起的所有 object
            objects_labels = batch_objects_labels[cur_index:cur_index+1][0]            
            for item_3 in zip(objects,objects_labels): # 从上述的 objects 以及labels 中对应取出单个                
                cur_dict = {} # cur_dict 都会被放入到spo_list 中
                obj , obj_label = item_3
                if (subject == obj): # 说明预测的subject 和 object 一样，这样的数据没有意义
                    continue
                val_1 = {} # 存放object
                val_2 = {} # 存放object_type
                cur_dict['predicate'] = batch_relations[cnt]                
                
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
    


if __name__ == "__main__":
    get_precision_recall_f1(golden_file="./data/dev_data.json",
                            predict_file="./data/predictions.json.zip")
