from models import RelationModel, SubjectModel,ObjectModel
import logging
import argparse
import os
import random
import time
import math
import json
from functools import partial
import codecs
import zipfile
import re
from tqdm import tqdm
import sys

import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler 
from torch.utils.data import BatchSampler,SequentialSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import BertTokenizer, BertForSequenceClassification

from data_loader import DuIEDataset, DataCollator, ObjectDataset,SubjectDataset,SubjectDataCollator
from data_loader import PredictSubjectDataset,PredictSubjectDataCollator
from utils import decode_subject,decode_object, decoding, find_entity, get_precision_recall_f1, write_prediction_results
from utils import decode_relation_class,post_process
from data_loader import from_dict,from_dict2_relation

parser = argparse.ArgumentParser()
parser.add_argument("--do_train", action='store_true', default=False, help="do train")
parser.add_argument("--do_predict", action='store_true', default=False, help="do predict")
parser.add_argument("--init_checkpoint", default=None, type=str, required=False, help="Path to initialize params from")
parser.add_argument("--data_path", default="./data", type=str, required=False, help="Path to data.")
parser.add_argument("--predict_data_file", default="./data/test_data.json", type=str, required=False, help="Path to data.")
parser.add_argument("--output_dir", default="./checkpoints", type=str, required=False, help="The output directory where the model_subject predictions and checkpoints will be written.")
parser.add_argument("--max_seq_length", default=128, type=int,help="The maximum total input sequence length after tokenization. Sequences longer "
    "than this will be truncated, sequences shorter will be padded.", )
parser.add_argument("--batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.", )
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--num_train_epochs", default=12, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--warmup_ratio", default=0, type=float, help="Linear warmup over warmup_ratio * total_steps.")
parser.add_argument("--seed", default=42, type=int, help="random seed for initialization")
parser.add_argument("--n_gpu", default=1, type=int, help="number of gpus to use, 0 for cpu.")
args = parser.parse_args()
# yapf: enable



# Reads subject_map.
subject_map_path = os.path.join(args.data_path, "subject2id.json")
if not (os.path.exists(subject_map_path) and os.path.isfile(subject_map_path)):
    sys.exit("{} dose not exists or is not a file.".format(subject_map_path))
with open(subject_map_path, 'r', encoding='utf8') as fp:
    subject_map = json.load(fp)

# Reads object_map.
object_map_path = os.path.join(args.data_path, "object2id.json")
if not (os.path.exists(object_map_path) and os.path.isfile(object_map_path)):
    sys.exit("{} dose not exists or is not a file.".format(object_map_path))
with open(object_map_path, 'r', encoding='utf8') as fp:
    object_map = json.load(fp)


# Reads label_map.
relation_map_path = os.path.join(args.data_path, "predicate2id.json")
if not (os.path.exists(relation_map_path) and os.path.isfile(relation_map_path)):
    sys.exit("{} dose not exists or is not a file.".format(relation_map_path))
with open(relation_map_path, 'r', encoding='utf8') as fp:
    relation_map = json.load(fp)    

subject_class_num =  len(subject_map.keys()) # 得出subject的class num
object_class_num = len(object_map.keys())  # 得出object 的class num    
relation_class_num = len(relation_map.keys())# 得出 relation 的 个数



# Reads subject_map.
id2subject_map_path = os.path.join(args.data_path, "id2subject.json")
if not (os.path.exists(id2subject_map_path) and os.path.isfile(id2subject_map_path)):
    sys.exit("{} dose not exists or is not a file.".format(id2subject_map_path))
with open(id2subject_map_path, 'r', encoding='utf8') as fp:
    id2subject_map = json.load(fp)

# Reads object_map.
id2object_map_path = os.path.join(args.data_path, "id2object.json")
if not (os.path.exists(id2object_map_path) and os.path.isfile(id2object_map_path)):
    sys.exit("{} dose not exists or is not a file.".format(id2object_map_path))
with open(id2object_map_path, 'r', encoding='utf8') as fp:
    id2object_map = json.load(fp)


# Reads label_map.
id2relation_map_path = os.path.join(args.data_path, "id2relation.json")
if not (os.path.exists(id2relation_map_path) and os.path.isfile(id2relation_map_path)):
    sys.exit("{} dose not exists or is not a file.".format(id2relation_map_path))
with open(id2relation_map_path, 'r', encoding='utf8') as fp:
    id2relation_map = json.load(fp)    


def set_random_seed(seed):
    """sets random seed"""
    random.seed(seed)
    np.random.seed(seed)
    #t.seed(seed)  # 为什么torch 也要设置这个seed ？

logging.basicConfig(format='%(asctime)s - %(levelname)s -%(name)s - %(message)s',
                    datefmt='%m/%d%/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger("duie")



"""
可视化subject的预测
"""
def visualize_subject(file_path,origin_info,batch_subjects,batch_subject_labels):
    with open(file_path,'aw') as f:
        for item in (zip(origin_info,batch_subjects,batch_subject_labels)):
            origin , subjects,labels = item
            text = origin['text']
            f.write(text+"\n")
            for lines in zip(subjects,labels):
                subject,label = lines
                f.write(subject +"\t" +label+"\n")                
            f.write("==============================\n")    



"""
使用训练好的模型，进行预测。
01.这里使用的是预测的结果，而不是golden
02.这里的数据集统一叫做 dev_data.json
"""
def do_predict(model_subject_path,model_object_path,model_relation_path):
    # Does predictions.
    print("\n====================start predicting====================")
    name_or_path = "/home/lawson/pretrain/bert-base-chinese"
    model_subject = SubjectModel(name_or_path,768,out_fea=subject_class_num)    
    model_subject.load_state_dict(t.load(model_subject_path))
    model_subject = model_subject.cuda()    

    model_object = ObjectModel(name_or_path,768,object_class_num)        
    model_object = model_object.cuda()
    model_object.load_state_dict(t.load(model_object_path))


    model_relation = RelationModel(name_or_path,relation_class_num)
    model_relation = model_relation.cuda()
    model_relation.load_state_dict(t.load(model_relation_path))

    tokenizer = BertTokenizer.from_pretrained("/home/lawson/pretrain/bert-base-chinese")    
    # Loads dataset.
    dev_dataset = PredictSubjectDataset.from_file(
        os.path.join(args.data_path, 'train_data_2.json'),
        #os.path.join(args.data_path, 'dev_data.json'),
        tokenizer,
        args.max_seq_length,
        True
        )
    
    collator = PredictSubjectDataCollator()
    dev_data_loader = DataLoader(        
        dataset=dev_dataset,
        batch_size=args.batch_size,
        collate_fn=collator, # 重写一个 collator
        )

    model_subject.eval()
    model_object.eval()
    model_relation.eval()
    # 将subject的预测结果写到文件中
    #file_path = "./data/subject_predict.txt"    
    res = [] # 最后的预测结果
    invalid_num = 0 # 预测失败的个数
    with t.no_grad():
        for batch in tqdm(dev_data_loader):
            input_ids,token_type_ids,attention_mask, origin_info = batch
            # labels size = [batch_size,max_seq_length]
            logits_1 = model_subject(input_ids=input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask
                                    )
            #logits size [batch_size,max_seq_len,class_num]  
            # 得到预测到的 subject
            batch_subjects,batch_subject_labels = decode_subject(logits_1,id2subject_map,input_ids,tokenizer)
            #visualize_subject(file_path,origin_info,batch_subjects,batch_subject_labels)
            
            # 需要判断 batch_subjects 是空的情况，最好能够和普通subjects 一样处理        
            if(len(batch_subjects[0]) == 0):
                #print("----- 未预测到subject ----------")        
                invalid_num+=1
                continue

            # ====== 根据origin_info 得到 subtask 2 的训练数据 ==========
            # 这里的object_input_ids 的 size 不再是args.batch_size ，可能比这个稍大
            object_input_ids, object_token_type_ids,object_attention_mask, object_labels = from_dict(batch_subjects,
                                                                                                        origin_info,
                                                                                                        tokenizer,
                                                                                                        args.max_seq_length,
                                                                                                        True,
                                                                                                        )
            object_input_ids = t.tensor(object_input_ids).cuda()
            object_token_type_ids = t.tensor(object_token_type_ids).cuda()
            object_attention_mask = t.tensor(object_attention_mask).cuda()
            
            logits_2 = model_object(input_ids = object_input_ids,
                                    token_type_ids=object_token_type_ids,
                                    attention_mask=object_attention_mask
                                    )
            batch_objects,batch_object_labels = decode_object(logits_2,id2object_map,tokenizer,object_input_ids)

            if(len(batch_objects[0]) == 0):
                invalid_num+=1
                #print("----- 未预测到 object ----------")        
                continue
            # ====== 根据 subject + object 得到 subtask 3 的训练数据 ==========        
            relation_input_ids, relation_token_type_ids, relation_attention_mask, relation_labels = from_dict2_relation(batch_subjects,
            batch_objects,
            origin_info,
            tokenizer,
            args.max_seq_length
            )
            
            relation_input_ids = t.tensor(relation_input_ids).cuda()
            relation_token_type_ids = t.tensor(relation_token_type_ids).cuda()
            relation_attention_mask = t.tensor(relation_attention_mask).cuda()        

            # 这个模型直接得到loss
            out = model_relation(input_ids=relation_input_ids,
                                    token_type_ids=relation_token_type_ids,
                                    attention_mask=relation_attention_mask,
                                    labels = None                                
                                    )
            logits = out.logits # 输出最后的分类分数
            # size [batch_size, relation_class_num]

            batch_relations = decode_relation_class(logits,id2relation_map)

            # 得到最后的结果
            cur_res = post_process(batch_subjects, # 5
                        batch_subject_labels, # 5
                        batch_objects, # 5
                        batch_object_labels,# 5
                        batch_relations,
                        origin_info[0]['text']
            )        
            res.append(cur_res)

    # 写出最后的结果
    predict_file_path = os.path.join(args.data_path, 'train_data_2_predict.json') 
    #predict_file_path = os.path.join(args.data_path, 'dev_data_predict.json') 
    with open(predict_file_path,"w",encoding="utf-8") as f:
        for line in res:        
            json_str = json.dumps(line,ensure_ascii=False)
            json_str=json_str[1:-1]
            #print(json_str)
            f.write(json_str)
            f.write('\n')          
    print(f"未预测到的个数是：{invalid_num}")
    print("=====predicting complete=====")


if __name__ == "__main__":
    set_random_seed(args.seed)
    # 指定GPU设备    
    device = t.device("cuda" if t.cuda.is_available() and not args.n_gpu else "cpu")    

    if args.do_predict:
        model_subject_path = "./model_subject/model_subject_64236.pdparams"
        model_object_path = "./model_object/model_object_214120.pdparams"
        model_relation_path  = "./model_relation/model_relation_513888.pdparams"
        do_predict(model_subject_path,model_object_path,model_relation_path)
