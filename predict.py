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
from metric import cal_subject_metric, cal_subject_object_metric

import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler 
from torch.utils.data import BatchSampler,SequentialSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import BertTokenizerFast, BertForSequenceClassification

from data_loader import PredictSubjectDataset,PredictSubjectDataCollator, from_dict2object,get_negative_relation_data
from utils import  add_relation_of_country, decode_subject,decode_object, decoding,  find_entity, get_all_country, get_all_subjects, get_precision_recall_f1, post_process_2, visualize_subject_object, write_prediction_results
from utils import decode_relation_class,post_process,addBookName,visualize_subject
from data_loader import from_dict2object,from_dict2_relation

parser = argparse.ArgumentParser()
parser.add_argument("--do_train", action='store_true', default=False, help="do train")
parser.add_argument("--do_predict", action='store_true', default=False, help="do predict")
parser.add_argument("--init_checkpoint", default=None, type=str, required=False, help="Path to initialize params from")
parser.add_argument("--data_path", default="./data", type=str, required=False, help="Path to data.")
parser.add_argument("--dev_data_path", default="./data", type=str, required=False, help="Path to data.")
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
parser.add_argument("--model_subject_path",type=str,help="ds")
parser.add_argument("--model_object_path",type=str,help="ds")
parser.add_argument("--model_relation_path",type=str,help="ds")
args = parser.parse_args()
# yapf: enable


import time
curTime = time.strftime("%m%d_%H%M%S", time.localtime())
log_name = "predict" + curTime + '.log'
logging.basicConfig(format='%(asctime)s - %(levelname)s -%(name)s - %(message)s',
                    datefmt='%m/%d%/%Y %H:%M:%S',
                    level=logging.INFO,
                    filemode='w',
                    filename="/home/lawson/program/DuIE_py/log/" + log_name
                    )
logger = logging.getLogger("predict")


# Reads subject_map.
subject_map_path = os.path.join(args.data_path, "subject2id_1.json")
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

subject_class_num =  len(subject_map.keys()) # ??????subject???class num
object_class_num = len(object_map.keys())  # ??????object ???class num    
relation_class_num = len(relation_map.keys())# ?????? relation ??? ??????



# Reads subject_map.
id2subject_map_path = os.path.join(args.data_path, "id2subject_1.json")
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
id2relation_map_path = os.path.join(args.data_path, "id2predicate.json")
if not (os.path.exists(id2relation_map_path) and os.path.isfile(id2relation_map_path)):
    sys.exit("{} dose not exists or is not a file.".format(id2relation_map_path))
with open(id2relation_map_path, 'r', encoding='utf8') as fp:
    id2relation_map = json.load(fp)    


def set_random_seed(seed):
    """sets random seed"""
    random.seed(seed)
    np.random.seed(seed)
    #t.seed(seed)  # ?????????torch ??????????????????seed ???




"""
??????: ??????batch_size > 1 ?????????
"""
def predict_subject(model_subject_path,out_file_path):
    # Does predictions.
    print("\n====================start predicting / evaluating ====================")
    #name_or_path = "/pretrains/pt/chinese_RoBERTa-wwm-ext_pytorch"
    name_or_path = "/home/lawson/pretrain/bert-base-chinese"
    model_subject = SubjectModel(name_or_path,768,out_fea=subject_class_num)
    model_subject.load_state_dict(t.load(model_subject_path))
    model_subject = model_subject.cuda()

    #tokenizer = BertTokenizerFast.from_pretrained("/pretrains/pt/chinese_RoBERTa-wwm-ext_pytorch")
    tokenizer = BertTokenizerFast.from_pretrained("/home/lawson/pretrain/bert-base-chinese")
    # Loads dataset.
    dev_dataset = PredictSubjectDataset.from_file(                
        args.dev_data_path,
        tokenizer,
        args.max_seq_length,
        True
        )
    
    collator = PredictSubjectDataCollator()
    dev_data_loader = DataLoader(        
        dataset=dev_dataset,
        batch_size=args.batch_size,
        collate_fn=collator, # ???????????? collator
        )

    model_subject.eval()
    
    res = [] # ?????????????????????
    invalid_num = 0 # ?????????????????????
    all_subjects = []
    all_subject_labels = []
    with t.no_grad():        
        for batch in tqdm(dev_data_loader):
            # origin_info ????????????json???????????????
            input_ids,token_type_ids,attention_mask, batch_origin_info,offset_mapping = batch
            # labels size = [batch_size,max_seq_length]
            logits_1 = model_subject(input_ids=input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask
                                    )
            #logits size [batch_size,max_seq_len,class_num]  
            # ?????????????????? subject
            # temp = get_rid_of_number_in_str(origin_info[0]['text'])
            # origin_info[0]['text'] = temp
            batch_subjects,batch_subject_labels = decode_subject(logits_1,
            id2subject_map,
            input_ids,
            tokenizer,
            batch_origin_info,
            offset_mapping
            )
            
            # ????????????????????? => ?????????????????????????????????????????? subject 
            for item in zip(batch_origin_info,batch_subjects,batch_subject_labels):
                origin_info,subjects,labels = item
                target = addBookName(origin_info['text'])
                subjects.extend(target)
                labels.extend(["?????????"] * len(target))

            all_subjects.append(batch_subjects)
            all_subject_labels.append(batch_subject_labels)
            # ???????????? batch_subjects ???????????????????????????????????????subjects ????????????        
            if(len(batch_subjects[0]) == 0):
                #print("----- ????????????subject ----------")
                invalid_num+=1
                continue
        visualize_subject(out_file_path,all_subjects,all_subject_labels)
        print(f"???????????????subject ????????????{invalid_num}")



"""
???subject??????????????????object
"""
def predict_subject_object(model_subject_path,model_object_path):
    # Does predictions.
    print("\n====================start predicting / evaluating ====================")
    #name_or_path = "/pretrains/pt/chinese_RoBERTa-wwm-ext_pytorch"
    subject_name_or_path = "/home/lawson/pretrain/bert-base-chinese"
    model_subject = SubjectModel(subject_name_or_path,768,out_fea=subject_class_num)
    model_subject.load_state_dict(t.load(model_subject_path))
    model_subject = model_subject.cuda()


    object_name_or_path = "/home/lawson/pretrain/bert-base-chinese"        
    model_object = ObjectModel(object_name_or_path,768,object_class_num)
    model_object = model_object.cuda()
    model_object.load_state_dict(t.load(model_object_path))
    
    tokenizer = BertTokenizerFast.from_pretrained("/home/lawson/pretrain/bert-base-chinese")    
    # Loads dataset.
    dev_dataset = PredictSubjectDataset.from_file(                
        args.dev_data_path,
        tokenizer,
        args.max_seq_length,
        True
        )
    
    collator = PredictSubjectDataCollator()
    dev_data_loader = DataLoader(        
        dataset=dev_dataset,
        batch_size=args.batch_size,
        collate_fn=collator, # ???????????? collator
        )

    model_subject.eval()
    model_object.eval()
    all_known_subjects = get_all_subjects("/home/lawson/program/DuIE_py/data/train_data.json")
    res = [] # ?????????????????????
    subject_invalid_num = 0 # ?????????????????????
    temp = (args.dev_data_path).split("/")[-1].split('.')[0]
    subject_object_predict_file =  f"./{temp}_subject_object_predict.txt"
    if os.path.exists(subject_object_predict_file):
        os.remove(subject_object_predict_file)
    with t.no_grad():        
        for batch in tqdm(dev_data_loader):
            # origin_info ????????????json???????????????
            input_ids,token_type_ids,attention_mask, batch_origin_info,offset_mapping = batch
            # labels size = [batch_size,max_seq_length]
            logits_1 = model_subject(input_ids=input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask
                                    )
            #logits size [batch_size,max_seq_len,class_num]  
            # ?????????????????? subject
            # temp = get_rid_of_number_in_str(origin_info[0]['text'])
            # origin_info[0]['text'] = temp
            batch_subjects,batch_subject_labels = decode_subject(logits_1,
            id2subject_map,
            input_ids,
            tokenizer,
            batch_origin_info,
            offset_mapping,
            all_known_subjects
            )            
                        
            
            logger.info("\n====================start predicting object ====================")                    
            # ???subject??????????????????????????????                        
            object_invalid_num = 0 
            # ====== ??????origin_info ?????? subtask 2 ??????????????? ==========
            # ?????????object_input_ids ??? size ?????????args.batch_size ????????????????????????
            object_input_ids, object_token_type_ids,object_attention_mask,\
            object_labels,object_origin_info,object_offset_mapping = from_dict2object(batch_subjects=batch_subjects,
                                                        batch_origin_dict=batch_origin_info,
                                                        tokenizer=tokenizer,
                                                        max_length=args.max_seq_length,
                                                        )
            object_input_ids = t.tensor(object_input_ids).cuda()
            object_token_type_ids = t.tensor(object_token_type_ids).cuda()
            object_attention_mask = t.tensor(object_attention_mask).cuda()
            
            logits_2 = model_object(input_ids = object_input_ids,
                                    token_type_ids=object_token_type_ids,
                                    attention_mask=object_attention_mask
                                    )
            batch_objects, batch_object_labels = decode_object(
                logits_2,
                id2object_map,
                tokenizer,
                object_input_ids,
                object_origin_info,
                object_offset_mapping,
                logger
            )

            # ?????????subject + object ???????????????
            visualize_subject_object(subject_object_predict_file,batch_subjects,batch_objects)            
    
    #??????
    cal_subject_object_metric(subject_object_predict_file,args.dev_data_path)
    
    

"""
??????????????????????????????????????????
01.?????????????????????????????????????????????golden
02.?????????????????????????????? dev_data.json
03.???????????????batch=1?????????
"""
def do_predict(model_subject_path,model_object_path,model_relation_path):
    # Does predictions.
    logger.info("\n====================start predicting====================")
    logger.info("\n===============????????????????????????????????????================")
    for k,v in (vars(args).items()):
        logger.info(f"{k,v}")
    bert_name_or_path = "/home/lawson/pretrain/bert-base-chinese"
    roberta_name_or_path = "/pretrains/pt/chinese_RoBERTa-wwm-ext_pytorch"    
    model_subject = SubjectModel(bert_name_or_path,768,out_fea=subject_class_num)
    #model_subject = SubjectModel(roberta_name_or_path,768,out_fea=subject_class_num)
    model_subject.load_state_dict(t.load(model_subject_path))
    model_subject = model_subject.cuda()    

    model_object = ObjectModel(bert_name_or_path,768,object_class_num)
    model_object = model_object.cuda()
    model_object.load_state_dict(t.load(model_object_path))

    model_relation = RelationModel(roberta_name_or_path,relation_class_num)
    model_relation = model_relation.cuda()
    model_relation.load_state_dict(t.load(model_relation_path))

    tokenizer = BertTokenizerFast.from_pretrained("/home/lawson/pretrain/bert-base-chinese")
    #predict_file_path = os.path.join(args.data_path, 'train_data_2_predict.json') 
    
    #subject_name = model_subject_path.    
    
    dev_data_path = (args.dev_data_path).split("/")[-1].split(".")[0]
    
    a = (args.model_relation_path).split("/")
    a = "_".join(a[-2::])
    a = a.split(".")[0]

    predict_file_path = os.path.join(args.data_path, dev_data_path)+f"_predict_{a}.json"
    batch_file_path = f"/home/lawson/program/DuIE_py/data/{dev_data_path}_subject_object_relation_{a}.txt"
    # Loads dataset.
    dev_dataset = PredictSubjectDataset.from_file(
        #os.path.join(args.data_path, 'train_data_2.json'),
        args.dev_data_path,
        tokenizer,
        args.max_seq_length,
        True
        )
    
    collator = PredictSubjectDataCollator()
    dev_data_loader = DataLoader(        
        dataset=dev_dataset,
        batch_size=args.batch_size,
        collate_fn=collator, # ???????????? collator
        )

    model_subject.eval()
    model_object.eval()
    model_relation.eval()

    all_known_subjects = get_all_subjects(train_data_path="/home/lawson/program/DuIE_py/data/train_data.json")
    # ???subject??????????????????????????????
    all_country = get_all_country(train_data_path="/home/lawson/program/DuIE_py/data/train_data.json")
    if os.path.exists(batch_file_path):
        logger.info("????????????subject_object_relation.txt????????????")
        sys.exit(0)
        
    res = [] # ?????????????????????
    invalid_num = 0 # ?????????????????????
    with t.no_grad():
        for batch in tqdm(dev_data_loader):
            # origin_info ????????????json???????????????
            input_ids,token_type_ids,attention_mask, batch_origin_info,offset_mapping = batch
            # labels size = [batch_size,max_seq_length]
            logits_1 = model_subject(input_ids=input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask
                                    )
            #logits size [batch_size,max_seq_len,class_num]  
            # ?????????????????? subject
            # temp = get_rid_of_number_in_str(origin_info[0]['text'])
            # origin_info[0]['text'] = temp
            batch_subjects,batch_subject_labels = decode_subject(logits_1,
            id2subject_map,
            input_ids,
            tokenizer,
            batch_origin_info,
            offset_mapping,
            all_known_subjects
            )
                        
            # ???subject??????????????????????????????                        
            object_invalid_num = 0 
            # ====== ??????origin_info ?????? subtask 2 ??????????????? ==========
            # ?????????object_input_ids ??? size ?????????args.batch_size ????????????????????????
            object_input_ids, object_token_type_ids,object_attention_mask,\
            object_labels,object_origin_info,object_offset_mapping = from_dict2object(batch_subjects=batch_subjects,
                                                        batch_origin_dict=batch_origin_info,
                                                        tokenizer=tokenizer,
                                                        max_length=args.max_seq_length,
                                                        )
            object_input_ids = t.tensor(object_input_ids).cuda()
            object_token_type_ids = t.tensor(object_token_type_ids).cuda()
            object_attention_mask = t.tensor(object_attention_mask).cuda()
            
            logits_2 = model_object(input_ids = object_input_ids,
                                    token_type_ids=object_token_type_ids,
                                    attention_mask=object_attention_mask
                                    )
            batch_objects, batch_object_labels = decode_object(
                logits_2,
                id2object_map,
                tokenizer,
                object_input_ids,
                object_origin_info,
                object_offset_mapping,
                logger
            )

            if(len(batch_objects[0]) == 0):
                invalid_num+=1
                #print("----- ???????????? object ----------")        
                continue
            # ====== ?????? subject + object ?????? subtask 3 ??????????????? ==========        
            relation_input_ids, relation_token_type_ids,\
            relation_attention_mask, relation_labels = from_dict2_relation(batch_subjects,
                                                                               batch_objects,
                                                                               batch_origin_info,
                                                                               tokenizer,
                                                                               args.max_seq_length
                                                                               )
            
            relation_input_ids = t.tensor(relation_input_ids).cuda()
            relation_token_type_ids = t.tensor(relation_token_type_ids).cuda()
            relation_attention_mask = t.tensor(relation_attention_mask).cuda()        
            if relation_input_ids.size(0) == 0:
                continue
            
            # ????????????????????????loss
            out = model_relation(input_ids=relation_input_ids,
                                    token_type_ids=relation_token_type_ids,
                                    attention_mask=relation_attention_mask,
                                    labels = None                                
                                    )
            logits = out.logits # ???????????????????????????
            # size [batch_size, relation_class_num]

            batch_relations = decode_relation_class(logits,id2relation_map)

            # batch_relations = add_relation_of_country(batch_subjects,batch_subject_labels,
            # batch_objects,batch_object_labels,batch_relations,batch_origin_info)

            # ?????????????????????
            cur_res = post_process_2(batch_subjects, # 5                        
                        batch_objects, # 5                        
                        batch_relations,
                        batch_origin_info
            )
            res.extend(cur_res)

            # ???????????????????????????
            with open(batch_file_path,'a') as f:
                a = str(batch_subjects)
                b = str(batch_objects)
                c = str(batch_relations)
                f.write(a+"\n")
                f.write(b+"\n")
                f.write(c+"\n")
                f.write("\n")

    # ???????????????????????????
    with open(predict_file_path,"w",encoding="utf-8") as f:
        for line in res:        
            json_str = json.dumps(line,ensure_ascii=False)                        
            #print(json_str)
            f.write(json_str)
            f.write('\n')

    logger.info(f"???????????????????????????{invalid_num}")
    logger.info("=====predicting complete=====")


if __name__ == "__main__":

    set_random_seed(args.seed)
    # ??????GPU??????
    device = t.device("cuda" if t.cuda.is_available() and not args.n_gpu else "cpu")    

    if args.do_predict:
        model_subject_path = args.model_subject_path
        model_object_path = args.model_object_path
        model_relation_path = args.model_relation_path
        do_predict(model_subject_path,model_object_path,model_relation_path)
        #predict_subject_object(model_subject_path,model_object_path)