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

import collections
import json
import logging
import os
from re import sub
from typing import Optional, List, Union, Dict, get_origin
from dataclasses import dataclass

import numpy as np
import torch as t
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
from transformers import BertTokenizerFast

from extract_chinese_and_punct import ChineseAndPunctuationExtractor
import pickle

InputFeature = collections.namedtuple("InputFeature", [
    "input_ids", "seq_len", "tok_to_orig_start_index", "tok_to_orig_end_index",
    "labels"
])

SubjectInputFeature = collections.namedtuple("SubjectInputFeature", [
    "input_ids","token_type_ids","attention_mask","seq_len", "labels", "offset_mapping"
])


ObjectInputFeature = collections.namedtuple("ObjectInputFeature", [
    "input_ids","token_type_ids","attention_mask","labels","offset_mapping"
])

# TODO：弄清楚这里的语法是什么
RelationInputFeature = collections.namedtuple("RelationInputFeature", [
    "input_ids","token_type_ids","attention_mask","label"    
])


"""
功能：根据 预测到的 subjects 和 objects 得到 realtion 的负样本
"""
def get_negative_relation_data(batch_subjects,
              batch_objects,
              batch_origin_info,#[{...},{...}...{}]  
              tokenizer: BertTokenizerFast,
              max_length: Optional[int] = 512,                            
              ):
    with open("/home/lawson/program/DuIE_py/data/relation2id.json", 'r', encoding='utf8') as fp:
        relation_map = json.load(fp)
    chineseandpunctuationextractor = ChineseAndPunctuationExtractor()

    # 初始化赋空值
    input_ids, attention_mask, token_type_ids, labels,offset_mapping = (
        [] for _ in range(5))
    
    
    ''' example 中的是数据格式如下：
    {
    "text": "古往今来，能饰演古龙小说人物“楚留香”的，无一不是娱乐圈公认的美男子，2011年，36岁的张智尧在《楚留香新传》里饰演楚留香，依旧帅得让人无法自拔",
    "spo_list": [
        {
            "predicate": "主演",
            "object_type": {
                "@value": "人物"
            },
            "subject_type": "影视作品",
            "object": {
                "@value": "张智尧"
            },
            "subject": "楚留香新传"
        },
        {
            "predicate": "饰演",
            "object_type": {
                "inWork": "影视作品",
                "@value": "人物"
            },
            "subject_type": "娱乐人物",
            "object": {
                "inWork": "楚留香新传",
                "@value": "楚留香"
            },
            "subject": "张智尧"
        }
    ]
    }
    '''    
    assert len(batch_subjects) == len(batch_origin_info)  # 必须一一对应
    cur_index = 0 
    for item in zip(batch_origin_info,batch_subjects):
        example, subjects = item
        objects = batch_objects[cur_index:cur_index+len(subjects)]
        # 这里的example 是单条语句，需要使用for 循环，将其拼接成多条   
        # 先预处理，将一个example 变成(在其前追加subject+['SEP'])变为多个 example
        examples = process_negative_example_relation(subjects,objects,example)            
        cur_index += len(subjects)
        # 紧接着处理每个example
        for example in examples:
            input_feature = convert_example_to_relation_feature(
                example, tokenizer, chineseandpunctuationextractor,
                relation_map, max_length)
            
            # 得到所有的训练数据
            input_ids.append(input_feature.input_ids)            
            labels.append(input_feature.label)
            token_type_ids.append(input_feature.token_type_ids)
            attention_mask.append(input_feature.attention_mask)  
            #offset_mapping.append(input_feature.offset_mapping)
    return (input_ids,token_type_ids,attention_mask,labels)


def parse_label(spo_list, label_map, tokens, tokenizer):
    # 2 tags for each predicate + I tag + O tag
    num_labels = 2 * (len(label_map.keys()) - 2) + 2
    seq_len = len(tokens)
    # initialize tag
    labels = [[0] * num_labels for i in range(seq_len)]
    #  find all entities and tag them with corresponding "B"/"I" labels
    for spo in spo_list:
        for spo_object in spo['object'].keys():
            # assign relation label
            if spo['predicate'] in label_map.keys():
                # simple relation
                label_subject = label_map[spo['predicate']]
                label_object = label_subject + 55
                subject_tokens = tokenizer._tokenize(spo['subject'])
                object_tokens = tokenizer._tokenize(spo['object']['@value'])
            else:
                # complex relation
                label_subject = label_map[spo['predicate'] + '_' + spo_object]
                label_object = label_subject + 55
                subject_tokens = tokenizer._tokenize(spo['subject'])
                object_tokens = tokenizer._tokenize(spo['object'][spo_object])

            subject_tokens_len = len(subject_tokens)
            object_tokens_len = len(object_tokens)

            # assign token label
            # there are situations where s entity and o entity might overlap, e.g. xyz established xyz corporation
            # to prevent single token from being labeled into two different entity
            # we tag the longer entity first, then match the shorter entity within the rest text
            forbidden_index = None
            if subject_tokens_len > object_tokens_len:
                for index in range(seq_len - subject_tokens_len + 1):
                    if tokens[index:index +
                              subject_tokens_len] == subject_tokens:
                        labels[index][label_subject] = 1
                        for i in range(subject_tokens_len - 1):
                            labels[index + i + 1][1] = 1
                        forbidden_index = index
                        break

                for index in range(seq_len - object_tokens_len + 1):
                    if tokens[index:index + object_tokens_len] == object_tokens:
                        if forbidden_index is None:
                            labels[index][label_object] = 1
                            for i in range(object_tokens_len - 1):
                                labels[index + i + 1][1] = 1
                            break
                        # check if labeled already
                        elif index < forbidden_index or index >= forbidden_index + len(
                                subject_tokens):
                            labels[index][label_object] = 1
                            for i in range(object_tokens_len - 1):
                                labels[index + i + 1][1] = 1
                            break

            else:
                for index in range(seq_len - object_tokens_len + 1):
                    if tokens[index:index + object_tokens_len] == object_tokens:
                        labels[index][label_object] = 1
                        for i in range(object_tokens_len - 1):
                            labels[index + i + 1][1] = 1
                        forbidden_index = index
                        break

                for index in range(seq_len - subject_tokens_len + 1):
                    if tokens[index:index +
                              subject_tokens_len] == subject_tokens:
                        if forbidden_index is None:
                            labels[index][label_subject] = 1
                            for i in range(subject_tokens_len - 1):
                                labels[index + i + 1][1] = 1
                            break
                        elif index < forbidden_index or index >= forbidden_index + len(
                                object_tokens):
                            labels[index][label_subject] = 1
                            for i in range(subject_tokens_len - 1):
                                labels[index + i + 1][1] = 1
                            break

    # if token wasn't assigned as any "B"/"I" tag, give it an "O" tag for outside
    for i in range(seq_len):
        if labels[i] == [0] * num_labels:
            labels[i][0] = 1

    return labels



'''
功能：获取subtask 1 中的label
01. 这个 subtask 1 就是一个普通的 NER任务
02. 为了更好地召回出subject，这里将object 和subject 都视为寻找的目标
params:
 subject_map：
 object_map: 
'''
def parse_subject_label(spo_list, subject_map, tokens, tokenizer,object_map):    
    # 将下面这些object数据排除掉，因为这些object 大都是谓语，不可能做subject 
    exclude = ['成立日期','获奖','上映时间','票房','海拔','修业年限','人口数量','面积','注册资本','邮政编码','占地面积','专业代码']
    seq_len = len(tokens)
    # initialize tag
    labels = [0 for i in range(seq_len)]
    #  find all entities and tag them with corresponding "B"/"I" labels
    for spo in spo_list:
        subject_val = spo['subject']  # 邪少兵王
        subject_val_tokens = tokenizer.tokenize(subject_val)
        subject_len = len(subject_val_tokens)
        subject_type = spo['subject_type']
        
        # 遍历找出下标
        # TODO:这里其实存在一个问题，就是如果 subject 在text中多次出现，那么该怎么办？ => 这里的处理是break
        for i,word in enumerate(tokens): 
            if tokens[i:i+subject_len] == subject_val_tokens:
                labels[i] = subject_map[subject_type] # 'B_图书作品'
                for j in range(i+1,i+subject_len):
                    labels[j] = 1 # 'I'
                break

        # ========== 找 object 的标签 ==========
        # predicate = spo['predicate']
        # if predicate not in exclude:            
        #     object_dict = spo['object']  # a dict
        #     objects = list(object_dict.values()) # 装入当前这个 spo 中的所有object        
        #     object_types = list(spo['object_type'].values())
        #     for item in zip(objects,object_types):
        #         object,object_type = item
        #         object_val_tokens = tokenizer.tokenize(object)
        #         object_len = len(object_val_tokens)                      
        #         # 遍历找出下标
        #         # TODO:这里其实存在一个问题，就是如果 subject 在text中多次出现，那么该怎么办？ => 照标不误
        #         for i,word in enumerate(tokens): 
        #             if tokens[i:i+object_len] == object_val_tokens:
        #                 # 如果object_type 出现在了subject_map 中，那么就继续使用subject_map中的标签，否则使用19
        #                 if object_type not in subject_map.keys():
        #                     labels[i] = 19  # 在 subject_map 之外的数据，统统设置为19
        #                 else:
        #                     labels[i] = subject_map[object_type]
        #                 for j in range(i+1,i+object_len):
        #                     labels[j] = 1 # 'I'
        #                 break
    return labels


'''
功能：解析object 的label。这个函数的目的是为了找到object 值的label
'''
def parse_object_label(spo_list, object_map, tokens, tokenizer):    
    seq_len = len(tokens)
    # initialize tag
    labels = [0 for i in range(seq_len)]
    include = []
    # 找出所有的object的值
    for spo in spo_list:
        object_vals = list(spo['object'].values())
        # 因为object_type 中的内容可能是复杂的结构，所以这里是 object_type_vals
        object_type_vals = list(spo['object_type'].values())
        object_val_tokens = []
        
        for object_val in object_vals:
            object_val_tokens.append(tokenizer.tokenize(object_val))
        
        # 找出有几个object
        for i,cur_object_type in enumerate(object_type_vals):
            cur_object_len = len(object_val_tokens[i])
            cur_object_val = object_val_tokens[i]
            for j,word in enumerate(tokens):                 
                if tokens[j:j+cur_object_len] == cur_object_val:
                    labels[j] = object_map[cur_object_type]
                    for k in range(j+1,j+cur_object_len):
                        labels[k] = 1 # 'I'
                    break
    
    # 如果把subject的值作为一部分训练数据，其实可能会扰乱分布
    # # 找出所有 subject 的值
    # for spo in spo_list:
    #     subject_val = spo['subject']    
    #     subject_type = spo['subject_type']
    #     if subject_type not in include:
    #         continue
    #     subject_val_tokens = []
    #     subject_val_tokens = tokenizer.tokenize(subject_val)
    #     cur_subject_len = len(subject_val_tokens)
    #     # 找出有几个object
    #     for j,word in enumerate(tokens):
    #         if tokens[j:j+cur_subject_len] == subject_val_tokens:
    #             labels[j] = object_map[subject_type]
    #             for k in range(j+1,j+cur_subject_len):
    #                 labels[k] = 1 # 'I'
    #             break
    return labels


'''
功能：解析 relation 的label。这个函数的目的是为了找到object 值的label
'''
def parse_relation_label(spo_list, relation_map):
    # initialize tag
    label = 0 # 默认分类为零
    # 再用这个做一个简单的分类就可以了，所以这里的lable就是一个数字    
    relation_vals = spo_list['predicate']
    label = relation_map[relation_vals]
    return label


def convert_example_to_feature(
        example,
        tokenizer:BertTokenizerFast,
        chineseandpunctuationextractor: ChineseAndPunctuationExtractor,
        label_map,
        max_length: Optional[int]=512,
        pad_to_max_length: Optional[bool]=None):
    spo_list = example['spo_list'] if "spo_list" in example.keys() else None
    text_raw = example['text']

    sub_text = []
    buff = ""
    for char in text_raw:
        if chineseandpunctuationextractor.is_chinese_or_punct(char):
            if buff != "":
                sub_text.append(buff)
                buff = ""
            sub_text.append(char)
        else:
            buff += char
    if buff != "":
        sub_text.append(buff)

    tok_to_orig_start_index = []
    tok_to_orig_end_index = []
    orig_to_tok_index = []
    tokens = []
    text_tmp = ''
    for (i, token) in enumerate(sub_text):
        orig_to_tok_index.append(len(tokens))
        sub_tokens = tokenizer._tokenize(token)
        text_tmp += token
        for sub_token in sub_tokens:
            tok_to_orig_start_index.append(len(text_tmp) - len(token))
            tok_to_orig_end_index.append(len(text_tmp) - 1)
            tokens.append(sub_token)
            if len(tokens) >= max_length - 2:
                break
        else:
            continue
        break

    seq_len = len(tokens)
    # 2 tags for each predicate + I tag + O tag
    num_labels = 2 * (len(label_map.keys()) - 2) + 2
    # initialize tag
    labels = [[0] * num_labels for i in range(seq_len)]
    if spo_list is not None:
        labels = parse_label(spo_list, label_map, tokens, tokenizer)

    # add [CLS] and [SEP] token, they are tagged into "O" for outside
    if seq_len > max_length - 2:
        tokens = tokens[0:(max_length - 2)]
        labels = labels[0:(max_length - 2)]
        tok_to_orig_start_index = tok_to_orig_start_index[0:(max_length - 2)]
        tok_to_orig_end_index = tok_to_orig_end_index[0:(max_length - 2)]
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    # "O" tag for [PAD], [CLS], [SEP] token
    outside_label = [[1] + [0] * (num_labels - 1)]

    labels = outside_label + labels + outside_label
    tok_to_orig_start_index = [-1] + tok_to_orig_start_index + [-1]
    tok_to_orig_end_index = [-1] + tok_to_orig_end_index + [-1]
    if seq_len < max_length:
        tokens = tokens + ["[PAD]"] * (max_length - seq_len - 2)
        labels = labels + outside_label * (max_length - len(labels))
        tok_to_orig_start_index = tok_to_orig_start_index + [-1] * (
            max_length - len(tok_to_orig_start_index))
        tok_to_orig_end_index = tok_to_orig_end_index + [-1] * (
            max_length - len(tok_to_orig_end_index))

    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    return InputFeature(
        input_ids=np.array(token_ids),
        seq_len=np.array(seq_len),
        tok_to_orig_start_index=np.array(tok_to_orig_start_index),
        tok_to_orig_end_index=np.array(tok_to_orig_end_index),
        labels=np.array(labels)
        )


"""
功能：获取subject 任务的feature
"""
def convert_example_to_subject_feature(
        example,
        tokenizer:BertTokenizerFast,
        chineseandpunctuationextractor: ChineseAndPunctuationExtractor,
        subject_map,
        object_map,
        max_length: Optional[int]=512,
        pad_to_max_length: Optional[bool]=None,
        is_train = False # 是否在训练模式
        ):
    spo_list = example['spo_list'] if "spo_list" in example.keys() else None
    text_raw = example['text']

    sub_text = []
    buff = ""
    for char in text_raw:
        if chineseandpunctuationextractor.is_chinese_or_punct(char):
            if buff != "":
                sub_text.append(buff)
                buff = ""
            sub_text.append(char)
        else:
            buff += char
    if buff != "":
        sub_text.append(buff)

    tokens = []
    text_tmp = ''    
    for (i, token) in enumerate(sub_text):    
        sub_tokens = tokenizer.tokenize(token)
        text_tmp += token
        for sub_token in sub_tokens:
            tokens.append(sub_token)
            if len(tokens) >= max_length - 2:
                break
        else:
            continue
        break

    seq_len = len(tokens)
    token_type_ids = [0] * seq_len
    attention_mask = [1] * seq_len    
    # initialize tag
    # 这里的labels 就是一个一维数组
    labels = [0 for i in range(seq_len)] 
    if spo_list is not None and is_train: 
        labels = parse_subject_label(spo_list, subject_map, tokens, tokenizer,object_map)

    # add [CLS] and [SEP] token, they are tagged into "O" for outside
    if seq_len > max_length - 2: # 这里的逻辑就是：如果seq_len 超过了最大长度，就截断
        tokens = tokens[0:(max_length - 2)]
        labels = labels[0:(max_length - 2)]
        token_type_ids = token_type_ids[0:max_length ] # 因为不用补充，所以不用减2
        attention_mask = attention_mask[0:max_length ]

    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    # "O" tag for [PAD], [CLS], [SEP] token    
    labels = [0] + labels + [0]
    token_type_ids = [0] + token_type_ids + [0]
    attention_mask = [1] + attention_mask + [1]
    # 如果长度不够，则调整
    while (len(tokens) < max_length ):
        tokens.append("[PAD]") # 追加
        labels.append(0)
        token_type_ids.append(0)
        attention_mask.append(0) 
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    temp = tokenizer(text_raw,
                    return_tensors="pt",
                    return_offsets_mapping= True,
                    max_length=128,                     
                    padding="max_length",
                    truncation=True
                    )
    offset_mapping = temp["offset_mapping"] # 得到offset 
    offset_mapping.squeeze_()
    # 为什么这里喜欢用np.array?
    return SubjectInputFeature(
        input_ids=np.array(token_ids),
        token_type_ids = np.array(token_type_ids),
        attention_mask = np.array(attention_mask),
        seq_len=np.array(seq_len),
        labels=np.array(labels),
        offset_mapping = np.array(offset_mapping)
        )


"""
功能：获取object 任务的feature
"""
def convert_example_to_object_feature(
        example,
        tokenizer:BertTokenizerFast,
        chineseandpunctuationextractor: ChineseAndPunctuationExtractor,
        subject_map,
        max_length: Optional[int]=512        
        ):
    spo_list = example['spo_list'] if "spo_list" in example.keys() else None
    text_raw = example['text']

    sub_text = []
    buff = ""
    for char in text_raw:
        if chineseandpunctuationextractor.is_chinese_or_punct(char):
            if buff != "":
                sub_text.append(buff)
                buff = ""
            sub_text.append(char)
        else:
            buff += char
    if buff != "":
        sub_text.append(buff)
    
    tokens = []
    text_tmp = ''    
    for (i, token) in enumerate(sub_text):
        sub_tokens = tokenizer.tokenize(token)
        text_tmp += token
        for sub_token in sub_tokens:
            tokens.append(sub_token)
            if len(tokens) >= max_length - 2:
                break
        else:
            continue
        break

    seq_len = len(tokens)
    token_type_ids = [0] * seq_len
    attention_mask = [1] * seq_len    
    # initialize tag
    # 这里的labels 就是一个一维数组
    labels = [0 for i in range(seq_len)] 
    if spo_list is not None: 
        labels = parse_object_label(spo_list, subject_map, tokens, tokenizer)

    # add [CLS] and [SEP] token, they are tagged into "O" for outside
    if seq_len > max_length - 2: # 这里的逻辑就是：如果seq_len 超过了最大长度，就截断
        tokens = tokens[0:(max_length - 2)]
        labels = labels[0:(max_length - 2)]
        token_type_ids = token_type_ids[0:max_length ] # 因为不用补充，所以不用减2
        attention_mask = attention_mask[0:max_length ]

    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    # "O" tag for [PAD], [CLS], [SEP] token    
    labels = [0] + labels + [0]
    token_type_ids = [0] + token_type_ids + [0]
    attention_mask = [1] + attention_mask + [1]
    # 如果长度不够，则调整
    while (len(tokens) < max_length ):
        tokens.append("[PAD]") # 追加
        labels.append(0)
        token_type_ids.append(0)
        attention_mask.append(0) 
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    temp = tokenizer(text_raw,
                    return_tensors="pt",
                  return_offsets_mapping= True,
                  max_length=128,
                  padding="max_length",
                  )
    offset_mapping = temp["offset_mapping"]
    offset_mapping.squeeze_()
    return ObjectInputFeature(
        input_ids=np.array(token_ids),
        token_type_ids = np.array(token_type_ids),
        attention_mask = np.array(attention_mask),        
        labels=np.array(labels),        
        offset_mapping=np.array(offset_mapping)
        )



def convert_example_to_relation_feature(
        example,
        tokenizer:BertTokenizerFast,
        chineseandpunctuationextractor: ChineseAndPunctuationExtractor,
        relation_map,
        max_length: Optional[int]=512,        
        ):
    spo_list = example['spo_list'] if "spo_list" in example.keys() else None
    text_raw = example['text']

    sub_text = []
    buff = ""
    for char in text_raw:
        if chineseandpunctuationextractor.is_chinese_or_punct(char):
            if buff != "":
                sub_text.append(buff)
                buff = ""
            sub_text.append(char)
        else:
            buff += char
    if buff != "":
        sub_text.append(buff)
    
    tokens = []
    text_tmp = ''    
    for (i, token) in enumerate(sub_text):
        sub_tokens = tokenizer.tokenize(token)
        text_tmp += token
        for sub_token in sub_tokens:
            tokens.append(sub_token)
            if len(tokens) >= max_length - 2:
                break
        else:
            continue
        break

    seq_len = len(tokens)
    token_type_ids = [0] * seq_len
    attention_mask = [1] * seq_len    
    # initialize tag
    # 这里的labels 就是一个一维数组
    label = [0 for i in range(seq_len)]
    if spo_list is not None : 
        label = parse_relation_label(spo_list, relation_map)

    # add [CLS] and [SEP] token, they are tagged into "O" for outside
    if seq_len > max_length - 2: # 这里的逻辑就是：如果seq_len 超过了最大长度，就截断
        tokens = tokens[0:(max_length - 2)]        
        token_type_ids = token_type_ids[0:max_length ] # 因为不用补充，所以不用减2
        attention_mask = attention_mask[0:max_length ]

    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    # "O" tag for [PAD], [CLS], [SEP] token    
    
    token_type_ids = [0] + token_type_ids + [0]
    attention_mask = [1] + attention_mask + [1]
    # 如果长度不够，则调整
    while (len(tokens) < max_length ):
        tokens.append("[PAD]") # 追加    
        token_type_ids.append(0)
        attention_mask.append(0)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    # temp = tokenizer(text_raw,
    #                 return_tensors="pt",
    #               return_offsets_mapping= True,
    #               max_length=128)
    # offset_mapping = temp["offset_mapping"]
    # offset_mapping.squeeze_()
    return RelationInputFeature(
        input_ids=np.array(token_ids),
        token_type_ids = np.array(token_type_ids),
        attention_mask = np.array(attention_mask),        
        label = label,
    #    offset_mapping= np.array(offset_mapping)
        )



@dataclass
class DataCollator:
    """
    Collator for DuIE.
    """

    def __call__(self, examples: List[Dict[str, Union[list, np.ndarray]]]):
        batched_input_ids = np.stack([x['input_ids'] for x in examples])
        seq_lens = np.stack([x['seq_lens'] for x in examples])
        tok_to_orig_start_index = np.stack(
            [x['tok_to_orig_start_index'] for x in examples])
        tok_to_orig_end_index = np.stack(
            [x['tok_to_orig_end_index'] for x in examples])
        labels = np.stack([x['labels'] for x in examples])

        return (batched_input_ids, seq_lens, tok_to_orig_start_index,
                tok_to_orig_end_index, labels)

# predict 模式下的 SubjectDataCollator
@dataclass
class SubjectDataCollator:
    """
    Collator for DuIE.
    """
    def __call__(self, examples):
        batched_input_ids = np.stack([x['input_ids'] for x in examples])        
        attention_mask = np.stack(
            [x['attention_mask'] for x in examples])
        token_typed_ids = np.stack(
            [x['token_type_ids'] for x in examples])
        labels = np.stack([x['labels'] for x in examples])
        origin_info = np.stack(x['origin_info'] for x in examples)
        batched_input_ids = t.tensor(batched_input_ids).cuda()
        token_typed_ids = t.tensor(token_typed_ids).cuda()
        attention_mask = t.tensor(attention_mask).cuda()        
        labels = t.tensor(labels).cuda()

        return (batched_input_ids,token_typed_ids,attention_mask, labels,origin_info)


# train 模式下的subject Data collator
@dataclass
class TrainSubjectDataCollator:
    """
    Collator for DuIE.
    """
    def __call__(self, examples):
        batched_input_ids = np.stack([x['input_ids'] for x in examples])        
        attention_mask = np.stack(
            [x['attention_mask'] for x in examples])
        token_typed_ids = np.stack(
            [x['token_type_ids'] for x in examples])
        labels = np.stack([x['labels'] for x in examples])     
        origin_info = np.stack(x['origin_info'] for x in examples)   
        offset_mapping = np.stack([x['offset_mapping'] for x in examples])


        batched_input_ids = t.tensor(batched_input_ids).cuda()
        token_typed_ids = t.tensor(token_typed_ids).cuda()
        attention_mask = t.tensor(attention_mask).cuda()        
        labels = t.tensor(labels).cuda()

        return (batched_input_ids,token_typed_ids,attention_mask,origin_info, labels,offset_mapping)


@dataclass
class TrainObjectDataCollator:
    """
    Collator for DuIE.
    """
    def __call__(self, examples):
        batched_input_ids = np.stack([x['input_ids'] for x in examples])        
        attention_mask = np.stack(
            [x['attention_mask'] for x in examples])
        token_typed_ids = np.stack(
            [x['token_type_ids'] for x in examples])
        labels = np.stack([x['labels'] for x in examples])     
        origin_info = np.stack(x['origin_info'] for x in examples)   
        offset_mapping = np.stack([x['offset_mapping'] for x in examples])

        batched_input_ids = t.tensor(batched_input_ids).cuda()
        token_typed_ids = t.tensor(token_typed_ids).cuda()
        attention_mask = t.tensor(attention_mask).cuda()        
        labels = t.tensor(labels).cuda()

        return (batched_input_ids,token_typed_ids,attention_mask,origin_info, labels,offset_mapping)



@dataclass
class PredictSubjectDataCollator:
    """
    Collator for DuIE.
    """
    def __call__(self, examples):
        batched_input_ids = np.stack([x['input_ids'] for x in examples])        
        attention_mask = np.stack(
            [x['attention_mask'] for x in examples])
        token_typed_ids = np.stack(
            [x['token_type_ids'] for x in examples])
        
        origin_info = np.stack(x['origin_info'] for x in examples)
        offset_mapping = np.stack([x['offset_mapping'] for x in examples])
        batched_input_ids = t.tensor(batched_input_ids).cuda()
        token_typed_ids = t.tensor(token_typed_ids).cuda()
        attention_mask = t.tensor(attention_mask).cuda()

        return (batched_input_ids,token_typed_ids,attention_mask, origin_info,offset_mapping)


"""
功能：这部分数据的加载 是为了预测 subject，仅是用于train部分，因为不想把 origin_info 数据也占用内存，所以这里
train  模式 和 predict 模式没有共用一个类

"""
class TrainSubjectDataset(Dataset):
    def __init__(
            self, 
            input_ids ,
            token_type_ids ,
            attention_mask,
            seq_lens,
            origin_info,
            labels,
            offset_mapping
            ):
        super(TrainSubjectDataset, self).__init__()

        self.input_ids = input_ids
        self.seq_lens = seq_lens        
        self.labels = labels
        self.token_type_ids = token_type_ids
        self.origin_info = origin_info
        self.attention_mask = attention_mask
        self.offset_mapping = offset_mapping

    def __len__(self):
        if isinstance(self.input_ids, np.ndarray):
            return self.input_ids.shape[0]
        else:
            return len(self.input_ids)

    def __getitem__(self, item):
        return {
            "input_ids": np.array(self.input_ids[item]),
            "token_type_ids":np.array(self.token_type_ids[item]),
            "attention_mask":np.array(self.attention_mask[item]),
            "seq_lens": np.array(self.seq_lens[item]),
            "origin_info":self.origin_info[item],
            # If model inputs is generated in `collate_fn`, delete the data type casting.
            "labels": np.array(self.labels[item], dtype=np.long),
            "offset_mapping":np.array(self.offset_mapping[item])
        }
    
    @classmethod
    def from_file(cls,
                  file_path: Union[str, os.PathLike],
                  tokenizer: BertTokenizerFast,
                  max_length: Optional[int]=512,
                  pad_to_max_length: Optional[bool]=None
                  ):
                  
        assert os.path.exists(file_path) and os.path.isfile(file_path), f"{file_path} dose not exists or is not a file."
        subject_map_path = os.path.join(os.path.dirname(file_path), "subject2id.json")
        assert os.path.exists(subject_map_path) and os.path.isfile(subject_map_path), f"{subject_map_path} dose not exists or is not a file."
        with open(subject_map_path, 'r', encoding='utf8') as fp:
            subject_map = json.load(fp)

        # Reads object_map.
        object_map_path = os.path.join(os.path.dirname(file_path), "object2id.json")        
        assert os.path.exists(object_map_path) and os.path.isfile(object_map_path),f"{object_map_path} dose not exists or is not a file."        
        with open(object_map_path, 'r', encoding='utf8') as fp:
            object_map = json.load(fp)

        chineseandpunctuationextractor = ChineseAndPunctuationExtractor()

        # 初始化赋空值
        input_ids, seq_lens, attention_mask, token_type_ids, labels, offset_mapping= (
            [] for _ in range(6))
        origin_info = [] # 原始文本信息
        dataset_scale = sum(1 for line in open(file_path, 'r'))        
        with open(file_path, "r", encoding="utf-8") as fp:
            lines = fp.readlines()
            for line in tqdm(lines):
                example = json.loads(line)
                input_feature = convert_example_to_subject_feature(
                    example, tokenizer, chineseandpunctuationextractor,
                    subject_map,object_map, max_length, pad_to_max_length,is_train=True)
                
                # 得到所有的训练数据
                input_ids.append(input_feature.input_ids)
                seq_lens.append(input_feature.seq_len)                
                labels.append(input_feature.labels)
                token_type_ids.append(input_feature.token_type_ids)
                attention_mask.append(input_feature.attention_mask)
                origin_info.append(example)
                offset_mapping.append(input_feature.offset_mapping)
        
        examples = cls(input_ids,
                       token_type_ids=token_type_ids,
                       attention_mask=attention_mask,
                       seq_lens=seq_lens,
                       origin_info = origin_info,
                       labels=labels,
                       offset_mapping = offset_mapping                   
                       )
        return examples


"""
功能：这部分数据的加载 是为了预测 subject 
"""
class PredictSubjectDataset(Dataset):    
    def __init__(
            self, 
            input_ids ,
            token_type_ids ,
            attention_mask,
            origin_info,
            offset_mapping
            ):
        super(PredictSubjectDataset, self).__init__()

        self.input_ids = input_ids        
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.origin_info = origin_info
        self.offset_mapping = offset_mapping

    def __len__(self):
        if isinstance(self.input_ids, np.ndarray):
            return self.input_ids.shape[0]
        else:
            return len(self.input_ids)

    def __getitem__(self, item):
        return {
            "input_ids": np.array(self.input_ids[item]),
            "token_type_ids":np.array(self.token_type_ids[item]),
            "attention_mask":np.array(self.attention_mask[item]),
            "origin_info":self.origin_info[item],
            "offset_mapping":np.array(self.offset_mapping[item])
        }
    
    @classmethod
    def from_file(cls,
                  file_path: Union[str, os.PathLike],
                  tokenizer: BertTokenizerFast,
                  max_length: Optional[int]=512,
                  pad_to_max_length: Optional[bool]=None
                  ):
        schema_file_path = "/home/lawson/program/DuIE_py/data/"        
        subject_map_path = os.path.join(os.path.dirname(schema_file_path), "subject2id.json")        
        assert os.path.exists(subject_map_path) and os.path.isfile(subject_map_path), f"{subject_map_path} dose not exists or is not a file."
        with open(subject_map_path, 'r', encoding='utf8') as fp:
            subject_map = json.load(fp)


        # Reads object_map.
        object_map_path = os.path.join(os.path.dirname(schema_file_path), "object2id.json")        
        assert os.path.exists(object_map_path) and os.path.isfile(object_map_path),f"{object_map_path} dose not exists or is not a file."        
        with open(object_map_path, 'r', encoding='utf8') as fp:
            object_map = json.load(fp)

        chineseandpunctuationextractor = ChineseAndPunctuationExtractor()

        # 初始化赋空值
        input_ids, seq_lens, attention_mask, token_type_ids, labels,offset_mapping = (
            [] for _ in range(6))
        origin_info = [] # 原始文本信息        
        print(f"Preprocessing data, loaded from %s" % file_path)
        with open(file_path, "r", encoding="utf-8") as fp:
            lines = fp.readlines()
            for line in tqdm(lines):
                example = json.loads(line)
                input_feature = convert_example_to_subject_feature(
                    example, tokenizer, chineseandpunctuationextractor,
                    subject_map,object_map, max_length, pad_to_max_length,is_train=False)
                
                # 得到所有的训练数据
                input_ids.append(input_feature.input_ids)
                seq_lens.append(input_feature.seq_len)                
                token_type_ids.append(input_feature.token_type_ids)
                attention_mask.append(input_feature.attention_mask)
                origin_info.append(example)
                offset_mapping.append(input_feature.offset_mapping)
        
        examples = cls(input_ids,
                       token_type_ids=token_type_ids,
                       attention_mask=attention_mask,
                       origin_info=origin_info,
                    offset_mapping= offset_mapping
                       )
        return examples




"""
功能：在一条训练样本（example） 中的 text 部分前添加 "subject + 。 " 得到 examples，
因为subject 可能有多个，所以这里最后的结果是 examples

01. example ： 是一条训练样本
02. subjects： 每条文本预测得到的 subject 的list 
"""
def process_example_object(example,subjects):
    examples = []
    spo_list = example['spo_list'] if "spo_list" in example.keys() else None
    text_raw = example['text']
    # 如果subjects 为None，说明是在train模式，那么我们自己手动生成，否则就用传入的subjects
    if subjects is None:
        subjects = []
        for spo in spo_list:
            subject = spo['subject']  # dict
            subjects.append(subject)
            text = subject + '。' + text_raw
            cur_example = {"spo_list":spo_list,"text":text}
            examples.append(cur_example)
    # in predict 
    elif len(subjects) == 0: # subject = [] 这种情况
        cur_example = {"spo_list":spo_list,"text":text_raw}
        examples.append(cur_example)
    else:
        # TODO 这里使用什么字符分割，也是一个待研究
        for subject in subjects:
            text = subject +'。'+ text_raw
            cur_example = {"spo_list":spo_list,"text":text}
            examples.append(cur_example)
    return examples


"""生成负样本
"""
def process_negative_example_relation(batch_subjects,batch_objects,example):
    examples = []
    text_raw = example['text']
    spo_list = example['spo_list']
    gold_subject_object_map = {} # 存储subject 和 object 的map
    gold_subject_object_label_map  = {} # subject 和 object 的label map
    for spo in spo_list:
        subjects = spo['subject']
        objects = list(spo['object'].values())
        if len(objects) == 1: # 简单结构
            if subjects == objects: # 为了避免二者相同
                continue
            if subjects not in gold_subject_object_map.keys():
                gold_subject_object_map[subjects] = []
                gold_subject_object_map[subjects].append(objects[0])
                
            else:
                gold_subject_object_map[subjects].append(objects[0])
            gold_subject_object_label_map[subjects+objects[0]] = spo['predicate']
        else: # 复杂结构
            for object in objects:
                if subjects == object: # 为了避免二者相同
                    continue
                if subjects not in gold_subject_object_map.keys():
                    gold_subject_object_map[subjects] = []
                    gold_subject_object_map[subjects].append(object)

                else:
                    gold_subject_object_map[subjects].append(object)
                gold_subject_object_label_map[subjects+object] = spo['predicate']

    for item in zip(batch_subjects,batch_objects):
        subject , objects = item
        for object in objects:
            text = subject + '。' + object +"。"+ text_raw
            if (subject not in gold_subject_object_map.keys()) or (object not in gold_subject_object_map[subject]): # 预测错误
                cur_example = {"spo_list":{"predicate":"O"},"text":text} 
                examples.append(cur_example)
            else:
                cur_example = {"spo_list":{"predicate":gold_subject_object_label_map[subject+object]},"text":text} 
                examples.append(cur_example)
    return examples







"""
功能：将 example 中的 text 文本前添加 subject + '。' + object + '。'  得到examples


params: 
 batch_objects 是一个[[],[],...]。 里面的每个都是 subject 的 预测集合

"""
def process_example_relation(batch_subjects,batch_objects,example):
    examples = []
    text_raw = example['text']

    # in train
    if batch_subjects is None and batch_objects is None:
        spo_list = example['spo_list'] if "spo_list" in example.keys() else None        
        
        # 每一条 spo 都会产生一个样本
        for spo in spo_list:
            subject = spo['subject']  # dict
            object_val = list(spo['object'].values())
            object_val = "。".join(object_val) + "。"        
            text = subject + '。' + object_val + text_raw
            cur_example = {"spo_list":spo,"text":text}
            examples.append(cur_example)
    
    # in predict
    else: # 这是个挺复杂的工作，因为subjects 和 objects 的值不是一一对应的
        # batch_subjects.squeeze_() # 首先给压平
        for item in zip(batch_subjects,batch_objects): # 找到每一个
                subject , objects = item
                for object in objects:                    
                    text = subject + '。' + object +"。"+ text_raw
                    # 因为是在predictr阶段，这里 的spo_list 是不用传入值的
                    cur_example = {"spo_list":None,"text":text} 
                    examples.append(cur_example)
    return examples


"""
train模式下 
使用 example 生成relation的训练数据
"""
def process_example_relation_train(example):
    examples = []
    text_raw = example['text']
    spo_list = example['spo_list'] if "spo_list" in example.keys() else None        
    
    # 每一条 spo 都会产生一个样本
    for spo in spo_list:
        subject = spo['subject']  # dict
        
        object_vals = list(spo['object'].values()) 
        object_types = list(spo['object_type'].values())
        if len(object_vals) > 1: # 复杂的结构体            
            for item in zip(object_vals,object_types):                
                cur_spo = spo
                cur_spo.pop("object_type")
                cur_spo.pop("object")
                object ,object_type = item
                object_val = "。" + object
                text = subject + '。' + object_val + text_raw
                cur_spo['object_type'] = {"@value":object_type}
                cur_spo['object'] = {"@value":object}
                cur_example = {"spo_list":cur_spo,"text":text}
                examples.append(cur_example)        
        else: # 简单结构
            object_val = "。".join(object_vals) + "。"
            text = subject + '。' + object_val + text_raw
            cur_example = {"spo_list":spo,"text":text}
            examples.append(cur_example)
    return examples




"""
从dict中构造object 需要的训练数据
"""
def from_dict2object(batch_subjects,
              batch_origin_dict,              
              tokenizer: BertTokenizerFast,
              max_length: Optional[int] = 512,
              pad_to_max_length = True
              ):

    with open("/home/lawson/program/DuIE_py/data/object2id.json", 'r', encoding='utf8') as fp:
        object_map = json.load(fp)
    chineseandpunctuationextractor = ChineseAndPunctuationExtractor()

    # 初始化赋空值  object_origin_info 是经过拼接subject后得到的 model_object 使用的原样本
    input_ids, attention_mask, token_type_ids, labels, object_origin_info,offset_mapping = (
        [] for _ in range(6))
    if batch_subjects is None: # in train
        for example in batch_origin_dict:            
            # 这里的example 是单条语句，需要使用for 循环，将其拼接成多条                
            # 先预处理，将一个example 变成(在其前追加subject+['SEP'])变为多个 example
            examples = process_example_object(example,subjects=None)
            for example in examples:
                input_feature = convert_example_to_object_feature(
                    example, tokenizer, chineseandpunctuationextractor,
                    object_map, max_length)
                
                # 得到所有的训练数据
                input_ids.append(input_feature.input_ids)            
                labels.append(input_feature.labels)
                token_type_ids.append(input_feature.token_type_ids)
                attention_mask.append(input_feature.attention_mask)  
                object_origin_info.append(example)
                offset_mapping.append(input_feature.offset_mapping)
        pass
    # in predict 
    else:
        assert len(batch_origin_dict) == len(batch_subjects)
        # batch_origin_dict 是原数据 [{},{} ... {}]
        for item in zip(batch_origin_dict,batch_subjects):
            example,subjects = item 
            # 这里的example 是原训练数据中的单条语句，需要使用for 循环，将其和预测得到的 subject 拼接成多条
            # 先预处理，将一个example 变成(在其前追加subject+['SEP'])变为多个 example
            examples = process_example_object(example,subjects)
            for example in examples:
                input_feature = convert_example_to_object_feature(
                    example, tokenizer, chineseandpunctuationextractor,
                    object_map, max_length )
                object_origin_info.append(example) # 加入原文本
                # 得到所有的训练数据
                input_ids.append(input_feature.input_ids)            
                labels.append(input_feature.labels)
                token_type_ids.append(input_feature.token_type_ids)
                attention_mask.append(input_feature.attention_mask)
                offset_mapping.append(input_feature.offset_mapping)                

    return (input_ids,token_type_ids,attention_mask,labels,object_origin_info,offset_mapping)


"""
功能：根据dict得到relation 的数据
batch_origin_dict [{},{}...{}]

01.
"""
def from_dict2_relation(batch_subjects,
              batch_objects,
              batch_origin_info,#[{...},{...}...{}]  
              tokenizer: BertTokenizerFast,
              max_length: Optional[int] = 512,                            
              ):
   
    with open("/home/lawson/program/DuIE_py/data/relation2id.json", 'r', encoding='utf8') as fp:
        relation_map = json.load(fp)
    chineseandpunctuationextractor = ChineseAndPunctuationExtractor()

    # 初始化赋空值
    input_ids, attention_mask, token_type_ids, labels,offset_mapping = (
        [] for _ in range(5))
    
    
    ''' example 中的是数据格式如下：
    {
    "text": "古往今来，能饰演古龙小说人物“楚留香”的，无一不是娱乐圈公认的美男子，2011年，36岁的张智尧在《楚留香新传》里饰演楚留香，依旧帅得让人无法自拔",
    "spo_list": [
        {
            "predicate": "主演",
            "object_type": {
                "@value": "人物"
            },
            "subject_type": "影视作品",
            "object": {
                "@value": "张智尧"
            },
            "subject": "楚留香新传"
        },
        {
            "predicate": "饰演",
            "object_type": {
                "inWork": "影视作品",
                "@value": "人物"
            },
            "subject_type": "娱乐人物",
            "object": {
                "inWork": "楚留香新传",
                "@value": "楚留香"
            },
            "subject": "张智尧"
        }
    ]
    }
    '''
    # train 模式  
    if batch_subjects is None and batch_objects is None:
        for example in batch_origin_info:            
            # 这里的example 是单条语句，需要使用for 循环，将其拼接成多条   
            # 先预处理，将一个example 变成(在其前追加subject+['SEP'])变为多个 example
            examples = process_example_relation_train(example)
            # 紧接着处理每个example
            for example in examples:
                input_feature = convert_example_to_relation_feature(
                    example, tokenizer, chineseandpunctuationextractor,
                    relation_map, max_length)
                
                # 得到所有的训练数据
                input_ids.append(input_feature.input_ids)            
                labels.append(input_feature.label)
                token_type_ids.append(input_feature.token_type_ids)
                attention_mask.append(input_feature.attention_mask)  
    else: # predict 模式
        assert len(batch_subjects) == len(batch_origin_info)  # 必须一一对应
        cur_index = 0
        for item in zip(batch_origin_info,batch_subjects):
            example, subjects = item
            if len(subjects) == 0:
                objects = batch_objects[cur_index:cur_index+1]
                cur_index += 1
            else:
                objects = batch_objects[cur_index:cur_index+len(subjects)]
                cur_index += len(subjects)
                
            # 这里的example 是单条语句，需要使用for 循环，将其拼接成多条   
            # 先预处理，将一个example 变成(在其前追加subject+['SEP'])变为多个 example
            examples = process_example_relation(subjects,objects,example)            
            
            # 紧接着处理每个example
            for example in examples:
                input_feature = convert_example_to_relation_feature(
                    example, tokenizer, chineseandpunctuationextractor,
                    relation_map, max_length)
                
                # 得到所有的训练数据
                input_ids.append(input_feature.input_ids)            
                labels.append(input_feature.label)
                token_type_ids.append(input_feature.token_type_ids)
                attention_mask.append(input_feature.attention_mask)  
                #offset_mapping.append(input_feature.offset_mapping)
    return (input_ids,token_type_ids,attention_mask,labels)


import chardet
# 写一个predicate2id 的脚本
def writePredictId(schema_path,out_path):
    with open(schema_path,'r',encoding='utf-8') as f:
        cont = json.load(f)
        predicate2id={}         
        print(len(cont))
        for block in cont:
            print(block)            
            predicate = block['predicate']
            temp = predicate.encode()
            print(chardet.detect(temp))
            if predicate not in predicate2id.keys():
                predicate2id[predicate] = len(predicate2id) +1        
        
    # 往predicate2id.json 文件写入
    with open(out_path,'w',encoding='utf-8') as f:
        # ensure_ascii 保证能够正常显示
        json.dump(predicate2id,f,ensure_ascii=False) 


if __name__ == "__main__":
    # tokenizer = ErnieTokenizer.from_pretrained("ernie-1.0")
    # d = DuIEDataset.from_file("./data/train_data.json", tokenizer)
    # sampler = torch.io.RandomSampler(data_source=d)
    # batch_sampler = torch.io.BatchSampler(sampler=sampler, batch_size=2)

    # collator = DataCollator()
    # loader = torch.io.DataLoader(
    #     dataset=d,
    #     batch_sampler=batch_sampler,
    #     collate_fn=collator,
    #     return_list=True)
    # for dd in loader():
    #     model_input = {
    #         "input_ids": dd[0],
    #         "seq_len": dd[1],
    #         "tok_to_orig_start_index": dd[2],
    #         "tok_to_orig_end_index": dd[3],
    #         "labels": dd[4]
    #     }
    #     print(model_input)
    path = "./data/duie_schema/duie_schema.json"
    out_path = './data/predicate2id_2.json'
    writePredictId(path,out_path)
