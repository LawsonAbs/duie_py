"""
训练relation 预测的模型

01.添加了O类预测。具体做法如下：
先在正样本上训练个 5 epoch，接着把预测失败的subject 和 object 作为负样本。 再训练 5epoch

"""
from torch.nn.modules.activation import LogSoftmax
from visdom import Visdom # 可视化输出loss
from models import RelationModel, SubjectModel,ObjectModel
import logging
import argparse
import os
import random
import time
import math
import json
from functools import partial, total_ordering
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
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers.utils.dummy_pt_objects import BertModel, get_polynomial_decay_schedule_with_warmup

from data_loader import  DataCollator,TrainSubjectDataset,TrainSubjectDataCollator, from_dict2_relation4_evaluate
from data_loader import PredictSubjectDataset,PredictSubjectDataCollator
from data_loader import get_negative_relation_data,from_dict2object
from utils import decode_relation_class, decode_subject,decode_object, decoding, find_entity, get_all_subjects, get_precision_recall_f1, post_process, write_prediction_results
from utils import addBookName
from data_loader import from_dict2_relation

parser = argparse.ArgumentParser()
parser.add_argument("--do_train", action='store_true', default=False, help="do train")
parser.add_argument("--do_eval", action='store_true', default=False, help="do predict")
parser.add_argument("--init_checkpoint", default=None, type=str, required=False, help="Path to initialize params from")
parser.add_argument("--data_path", default="./data", type=str, required=False, help="Path to data.")
parser.add_argument("--train_data_path", default="./data", type=str, required=False, help="Path to data.")

parser.add_argument("--model_subject_path", default="./data", type=str, required=False, help="Path to data.")
parser.add_argument("--model_object_path", default="./data", type=str, required=False, help="Path to data.")
parser.add_argument("--model_relation_path", default="./data", type=str, required=False, help="Path to data.")

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
args = parser.parse_args()
# yapf: enable



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

subject_class_num =  len(subject_map.keys()) # 得出subject的class num
object_class_num = len(object_map.keys())  # 得出object 的class num    
relation_class_num = len(relation_map.keys())# 得出 relation 的 个数



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

pred_file_path = "/home/lawson/program/DuIE_py/data//predict/relation/relation_predict.txt"

def set_random_seed(seed):
    """sets random seed"""
    random.seed(seed)
    np.random.seed(seed)
    #t.seed(seed)  # 为什么torch 也要设置这个seed ？

import time
curTime = time.strftime("%m%d_%H%M%S", time.localtime())
log_name = "model_relation_" + curTime + '.log'
logging.basicConfig(format='%(asctime)s - %(levelname)s -%(name)s - %(message)s',
                    datefmt='%m/%d%/%Y %H:%M:%S',
                    level=logging.INFO,
                    filemode='w',
                    filename="/home/lawson/program/DuIE_py/log/" + log_name
                    )
logger = logging.getLogger("relation")


# 评测函数
def evaluate(model_relation,dev_data_loader,tokenizer,pred_file_path):     
    model_relation.eval() # 根据subject+object 预测 relation    
    gold_num = 0
    correct_num = 0
    m = LogSoftmax(dim=-1)
    with t.no_grad():
        for batch in tqdm(dev_data_loader):
            batch_input_ids,batch_token_type_ids,batch_attention_mask, batch_origin_info,batch_labels,batch_offset_mapping = batch            
            # ====== 根据origin_info 得到 subtask 3 的训练数据 ==========
            # 根据 subject + object 预测 relation
            relation_input_ids, relation_token_type_ids, relation_attention_mask, relation_labels,batch_subjects,batch_objects = from_dict2_relation4_evaluate(batch_origin_info= batch_origin_info ,tokenizer=tokenizer,max_length=args.max_seq_length)
            
            relation_input_ids = t.tensor(relation_input_ids).cuda()
            relation_token_type_ids = t.tensor(relation_token_type_ids).cuda()
            relation_attention_mask = t.tensor(relation_attention_mask).cuda()            
            batch_relation_labels = t.tensor(relation_labels).cuda()
            # 这个模型直接得到loss
            out = model_relation(input_ids=relation_input_ids,
                                    token_type_ids=relation_token_type_ids,
                                    attention_mask=relation_attention_mask,
                                    labels = batch_relation_labels
                                    )                        


            # 计算metric
            logits = out.logits # 输出最后的分类分数
            # size [batch_size, relation_class_num]
            
            # a = m(logits)
            indexs = logits.argmax(-1).tolist() # 找出每个batch的属性下标

            # 计算出
            for item in zip(indexs,relation_labels): 
                pred,gold = item
                if pred == gold:
                    correct_num+=1
            gold_num += relation_input_ids.size(0)

            # 将最后的结果写入到 relation_predict.txt 中
            with open(pred_file_path,'a') as f:
                for item in zip(batch_subjects,batch_objects,indexs,relation_labels):
                    subject ,obj ,index ,label = item            
                    line = subject +"\t"+ obj+"\t" + id2relation_map[str(index)] +"\t" + id2relation_map[str(label)]+"\n"
                    if index == label:
                        line = "√\t" + line
                    else:
                        line = "x\t" + line
                    f.write(line)
    f1 = correct_num / gold_num
    # 计算 recall ,precision ,f1
    print(f"recall = {f1}, precision = {f1}, f1 = {f1}") 
    logger.info(f"recall = {f1}, precision = {f1}, f1 = {f1}") 
    logger.info("\n=====evaluating complete=====")


def do_train():
    # ========================== subtask 1. 预测subject ==========================
    name_or_path = "/pretrains/pt/chinese_RoBERTa-wwm-ext_pytorch"
    model_relation = RelationModel(name_or_path,relation_class_num)
    model_relation = model_relation.cuda()
    tokenizer = BertTokenizerFast.from_pretrained("/pretrains/pt/chinese_RoBERTa-wwm-ext_pytorch")    

    # Loads dataset.
    # 这里之所以使用 TrainSubjectDataset 是因为需要加载原始的数据，通过原始的数据才可以得到训练 relation 的数据
    train_dataset = TrainSubjectDataset.from_file(
        args.train_data_path,
        tokenizer,
        args.max_seq_length,
        True
        )
  
    collator = TrainSubjectDataCollator()
    train_data_loader = DataLoader(        
        dataset=train_dataset,
        #batch_sampler=train_batch_sampler,
        batch_size=args.batch_size,
        collate_fn=collator, # 重写一个 collator
        )
        
    dev_dataset = TrainSubjectDataset.from_file(
                                                args.dev_data_path,
                                                tokenizer,
                                                args.max_seq_length,
                                                True
                                                )
        
    dev_data_loader = DataLoader(
        dataset=dev_dataset,
        batch_size= args.batch_size,
        #batch_sampler=dev_batch_sampler,
        collate_fn=collator,
    )
    
    # 需要合并所有模型的参数    
    optimizer = t.optim.AdamW(
        [{'params':model_relation.parameters(),'lr':2e-5}        
        ],
        #weight_decay=args.weight_decay,
        ) 

    # Defines learning rate strategy.
    steps_by_epoch = len(train_data_loader)
    num_training_steps = steps_by_epoch * args.num_train_epochs    
    lr_scheduler = ReduceLROnPlateau(optimizer=optimizer,
                                     mode='min')
    
    # Starts training.
    global_step = 0
    logging_steps = 100    
    tic_train = time.time()
    viz = Visdom()
    win = "train_loss"
    for epoch in tqdm(range(args.num_train_epochs)):        
        print("\n=====start training of %d epochs=====" % epoch)
        tic_epoch = time.time()
        # 设置为训练模式    
        model_relation.train() # 根据subject+object 预测 relation
        logger_loss = 0
        step = 1
        for batch in tqdm(train_data_loader):
            batch_input_ids,batch_token_type_ids,batch_attention_mask, batch_origin_info,batch_labels,batch_offset_mapping = batch
            
            # ====== 根据origin_info 得到 subtask 3 的训练数据 ==========
            # 根据 subject + object 预测 relation
            relation_input_ids, relation_token_type_ids, relation_attention_mask, relation_labels = from_dict2_relation(batch_subjects=None,batch_objects=None, batch_origin_info= batch_origin_info ,tokenizer=tokenizer,max_length=args.max_seq_length)
            
            relation_input_ids = t.tensor(relation_input_ids).cuda()
            relation_token_type_ids = t.tensor(relation_token_type_ids).cuda()
            relation_attention_mask = t.tensor(relation_attention_mask).cuda()
            relation_labels = t.tensor(relation_labels).cuda()

            # 这个模型直接得到loss
            out = model_relation(input_ids=relation_input_ids,
                                 token_type_ids=relation_token_type_ids,
                                 attention_mask=relation_attention_mask,
                                 labels = relation_labels
                                 )
            loss = out.loss
            loss.backward()
            optimizer.step()
            #lr_scheduler.step()
            optimizer.zero_grad()
            loss_item = loss.item()            
            logger_loss += loss_item
            logger.info(f"loss:{loss_item}")

            # 打日志
            if global_step % logging_steps == 0 and global_step:
                logger.info(
                    f"epoch:{epoch}/{args.num_train_epochs},  steps:{step}/{steps_by_epoch},   loss:{loss_item},  speed: {logging_steps / (time.time() - tic_train)} step/s")
                tic_train = time.time()                
                viz.line([logger_loss], [global_step], win=win, update="append")
                logger_loss = 0
                           
            step+=1
            global_step += 1
        # 使用 dev 数据集评测模型效果
        evaluate(model_relation,dev_data_loader,tokenizer,pred_file_path)
        # 每个epoch之后保存模型
        logger.info(f"saving checkpoing model_relation_{global_step}.pdparams to {args.output_dir}")
        cur_model_name = os.path.join(args.output_dir,"model_relation_%d_roberta.pdparams" % (global_step))
        t.save(model_relation.state_dict(),cur_model_name)
        logger.info("\n=====training complete=====")


"""使用负样本进行训练，主要步骤如下:
01.使用 model_subject 和  model_object模型 生成subject 和 object
02.根据生成的 subject,object 对，去比对训练数据，增添负样本

"""
def do_train_2(model_subject_path,model_object_path,model_relation_path):
    # Does predictions.
    logger.info("\n====================start predicting / evaluating ====================")    
    subject_name_or_path = "/home/lawson/pretrain/bert-base-chinese"
    model_subject = SubjectModel(subject_name_or_path,768,out_fea=subject_class_num)
    model_subject.load_state_dict(t.load(model_subject_path))
    model_subject = model_subject.cuda()

    object_name_or_path = "/home/lawson/pretrain/bert-base-chinese" 
    model_object = ObjectModel(object_name_or_path,768,object_class_num)
    model_object = model_object.cuda()
    model_object.load_state_dict(t.load(model_object_path))
    
    relation_name_or_path = "/pretrains/pt/chinese_RoBERTa-wwm-ext_pytorch"
    model_relation = RelationModel(relation_name_or_path,relation_class_num)
    model_relation = model_relation.cuda()
    model_relation.load_state_dict(t.load(model_relation_path))
    tokenizer = BertTokenizerFast.from_pretrained("/pretrains/pt/chinese_RoBERTa-wwm-ext_pytorch")
    
    # Loads dataset.
    # 这里之所以使用 TrainSubjectDataset 是因为需要加载原始的数据，通过原始的数据才可以得到训练 relation 的数据
    logger.info(f"Preprocessing data, loaded from {args.train_data_path}")
    train_dataset = TrainSubjectDataset.from_file(
        args.train_data_path,
        tokenizer,
        args.max_seq_length,
        True
        )
    # 这里将DistributedBatchSample(paddle) 修改成了 DistributedSample(torch)    
    # 如果使用 DistributedSampler 那么应该就是一个多进程加载数据
    # train_batch_sampler = DistributedSampler(
    #     train_dataset,
    #     shuffle=True,
    #     drop_last=True 
    #     )
    collator = TrainSubjectDataCollator()
    train_data_loader = DataLoader(        
        dataset=train_dataset,
        #batch_sampler=train_batch_sampler,
        batch_size=args.batch_size,
        collate_fn=collator, # 重写一个 collator
        )
        

    model_subject.eval()
    viz = Visdom()
    win = "train_loss_negative_2"  
    res = [] # 最后的预测结果
    subject_invalid_num = 0 # 预测失败的个数
    all_known_subjects = get_all_subjects(train_data_path="/home/lawson/program/DuIE_py/data/train_data.json")
    # 将二者模型的梯度关闭
    for param in model_subject.parameters():
        param.requires_grad = False
    for param in model_object.parameters():
        param.requires_grad = False

    
    optimizer = t.optim.AdamW(
        [{'params':model_relation.parameters(),'lr':2e-5},
        ],
        )

    # Starts training.
    global_step = 0
    logging_steps = 50
    save_steps = 5000
    step = 1
    logging_loss = 0
    for epoch in tqdm(range(args.num_train_epochs)):
        total_neg_cnt = 0
        total_pos_cnt = 0
        for batch in tqdm(train_data_loader):
            # origin_info 是原始的json格式的信息
            input_ids,token_type_ids,attention_mask, batch_origin_info,batch_labels,offset_mapping = batch
            # labels size = [batch_size,max_seq_length]
            logits_1 = model_subject(input_ids=input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask
                                    )
            #logits size [batch_size,max_seq_len,class_num]
            # 得到预测到的 subject
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
            
            # 将subjects 中的元素去重                  
            # 需要判断 batch_subjects 是空的情况，最好能够和普通subjects 一样处理        
            if(len(batch_subjects[0]) == 0):
                #print("----- 未预测到subject ----------")
                subject_invalid_num+=1
                continue
                        
            # 将subject的预测结果写到文件中                        
            object_invalid_num = 0 
            # ====== 根据origin_info 得到 subtask 2 的训练数据 ==========
            # 这里的object_input_ids 的 size 不再是args.batch_size ，可能比这个稍大
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

            relation_input_ids,relation_token_type_ids,relation_attention_mask,relation_labels,batch_neg_cnt,batch_pos_cnt \
                = get_negative_relation_data(batch_subjects,batch_objects,batch_origin_info,tokenizer,max_length=128)
            relation_input_ids = t.tensor(relation_input_ids).cuda()
            relation_token_type_ids = t.tensor(relation_token_type_ids).cuda()
            relation_attention_mask = t.tensor(relation_attention_mask).cuda()
            relation_labels = t.tensor(relation_labels).cuda()
            
            if relation_input_ids.size(0) < 1:
                continue
            logger.info(f"relation_input_ids.size(0) = {relation_input_ids.size(0)}")
            if relation_input_ids.size(0) > 32:
                out = model_relation(input_ids=relation_input_ids[0:32,:],
                                        token_type_ids=relation_token_type_ids[0:32,:],
                                        attention_mask=relation_attention_mask[0:32,:],
                                        labels = relation_labels[0:32]
                                        )
                logger.info(f"{batch_origin_info}")
            else:
                # 这个模型直接得到loss
                out = model_relation(input_ids=relation_input_ids,
                                        token_type_ids=relation_token_type_ids,
                                        attention_mask=relation_attention_mask,
                                        labels = relation_labels
                                        )
            loss = out.loss
            loss.backward()
            optimizer.step()
            #lr_scheduler.step()
            optimizer.zero_grad()                                    
            
            if relation_input_ids.size(0) > 32:
                avg_loss = loss.item() / 32 
            else:
                avg_loss = loss.item() / relation_input_ids.size(0) 
            logger.info(f"平均每个样本的损失时：avg_loss = {avg_loss}")
            if avg_loss > 2: # 重点关注一下这种损失的数据
                logger.info(f"{batch_origin_info}")
            logging_loss += avg_loss
            # 打日志
            if global_step % logging_steps ==0 and global_step : 
                viz.line([logging_loss], [global_step], win=win, update="append")
                logging_loss = 0
            
            # 保存模型
            if global_step % save_steps == 0 and global_step != 0 :
                logger.info(f"saving checkpoing model_relation_{513882+global_step}.pdparams to {args.output_dir}")
                cur_model_name = os.path.join(args.output_dir,"model_relation_%d_roberta.pdparams" % (513882+global_step))
                t.save(model_relation.state_dict(),cur_model_name)

            total_neg_cnt += batch_neg_cnt
            total_pos_cnt += batch_pos_cnt  
            logger.info(f"batch_neg_cnt:{batch_neg_cnt}\n,\
                        batch_pos_cnt ={batch_pos_cnt}\n,\
                        total_neg_cnt={total_neg_cnt}\n,\
                        total_pos_cnt={total_pos_cnt}")
            step+=1
            global_step += 1
    
        # 每个epoch 之后保存模型
        t.save(model_relation.state_dict(),os.path.join(args.output_dir,
                        "model_relation_%d_roberta_epoch.pdparams" % (513882+global_step)))
    logger.info("\n=====training complete=====")



if __name__ == "__main__":
    set_random_seed(args.seed)
    # 指定GPU设备
    device = t.device("cuda" if t.cuda.is_available() and not args.n_gpu else "cpu")    

    if args.do_train:
        model_subject_path = args.model_subject_path
        model_object_path = args.model_object_path
        model_relation_path = args.model_relation_path
        do_train_2(model_subject_path,model_object_path,model_relation_path)
        #do_train()
    if args.do_eval:
        # ========================== subtask 1. 预测subject ==========================
        name_or_path = "/pretrains/pt/chinese_RoBERTa-wwm-ext_pytorch"
        model_relation = RelationModel(name_or_path,relation_class_num)
        
        tokenizer = BertTokenizerFast.from_pretrained("/pretrains/pt/chinese_RoBERTa-wwm-ext_pytorch")    
        if (args.model_relation_path != None) and os.path.exists(args.model_relation_path): # 加载初始模型
            model_relation.load_state_dict(t.load(args.model_relation_path))
        model_relation = model_relation.cuda()
        collator = TrainSubjectDataCollator()
        # load dataset
        dev_dataset = TrainSubjectDataset.from_file(
                                                args.dev_data_path,
                                                tokenizer,
                                                args.max_seq_length,
                                                True
                                                )
        
        dev_data_loader = DataLoader(
            dataset=dev_dataset,
            batch_size= args.batch_size,
            #batch_sampler=dev_batch_sampler,
            collate_fn=collator,
        )        
        if os.path.exists(pred_file_path):
            os.remove(pred_file_path)
        evaluate(model_relation,dev_data_loader,tokenizer,pred_file_path)