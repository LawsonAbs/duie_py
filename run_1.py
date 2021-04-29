"""
运行subject预测的模型
"""
from visdom import Visdom
import logging
from torchcrf import CRF
import visdom
from models import RelationModel, SubjectModel,ObjectModel
import argparse
import os
import random
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
from transformers import BertTokenizerFast, BertForSequenceClassification

from data_loader import DataCollator,TrainSubjectDataset,TrainSubjectDataCollator
from data_loader import PredictSubjectDataset,PredictSubjectDataCollator
from utils import decode_subject,decode_subject_2,decode_subject_3,decode_object, decode_subject_4, decode_subject_crf, decoding, find_entity, get_all_subjects, get_precision_recall_f1, visualize_subject_with_label, write_prediction_results, addBookName

from data_loader import from_dict2object,from_dict2_relation
from utils import visualize_subject
from metric import cal_subject_metric


parser = argparse.ArgumentParser()
parser.add_argument("--do_train", action='store_true', default=False, help="do train")
parser.add_argument("--do_eval", action='store_true', default=False, help="do train")
parser.add_argument("--do_predict", action='store_true', default=False, help="do predict")
parser.add_argument("--init_checkpoint", default=None, type=str, required=False, help="Path to initialize params from")
parser.add_argument("--data_path", default="./data", type=str, required=False, help="Path to data.")
parser.add_argument("--train_data_path", default="./data", type=str, required=False, help="Path to train data.")
parser.add_argument("--dev_data_path", default="./data", type=str, required=False, help="Path to dev data.")
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
#subject_class_num =  2
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


# 这没调用？
def set_random_seed(seed):
    """sets random seed"""
    random.seed(seed)
    np.random.seed(seed)
    # t.seed(seed)  # 为什么torch 也要设置这个seed ？

import time
curTime = time.strftime("%m%d_%H%M%S", time.localtime())
log_name = "model_subject_" + curTime + '.log'
logging.basicConfig(format='%(asctime)s - %(levelname)s -%(name)s - %(message)s',
                    datefmt='%m/%d%/%Y %H:%M:%S',
                    level=logging.INFO,
                    filename="/home/lawson/program/DuIE_py/log/" + log_name,
                    filemode='w', # 追加模式
                    )
logger = logging.getLogger("model_subject")

if args.do_train:
    vis = Visdom()
    win = "subject_loss"

tokenizer = BertTokenizerFast.from_pretrained("/pretrains/pt/chinese_RoBERTa-wwm-ext_pytorch")
criterion = nn.CrossEntropyLoss() # 使用交叉熵计算损失
collator = TrainSubjectDataCollator()

   

"""
功能： 评测部分
"""
def evaluate(model_subject,dev_data_loader,criterion,pred_file_path, crf,all_known_subjects):
    # Does predictions.
    logger.info("\n====================start  evaluating ====================")   
    tokenizer = BertTokenizerFast.from_pretrained("/pretrains/pt/chinese_RoBERTa-wwm-ext_pytorch")    
    # 将subject的预测结果写到文件中
    if os.path.exists(pred_file_path):# 因为下面是追加写入到文件中，所以
        os.remove(pred_file_path)
    total_loss = 0 # 总损失
    invalid_num = 0 # 预测失败的个数
    all_subjects = []
    all_subject_labels = []
    with t.no_grad():        
        for batch in tqdm(dev_data_loader):
            # origin_info 是原始的json格式的信息
            input_ids,token_type_ids,attention_mask, batch_origin_info,labels,offset_mapping = batch
            # labels size = [batch_size,max_seq_length]
            logits_1 = model_subject(input_ids=input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask
                                    )            
            #logits size [batch_size,max_seq_len,class_num]  
            #preds = crf.decode(logits_1)

            temp = logits_1.view(-1,subject_class_num)
            labels = labels.view(-1)
            cur_loss = criterion(temp, labels)
            total_loss += cur_loss
            # 得到预测到的 subject
            # temp = get_rid_of_number_in_str(origin_info[0]['text'])
            # origin_info[0]['text'] = temp
            #preds = t.tensor(preds).cuda()
            batch_subjects,batch_subject_labels = decode_subject(logits_1,
                                                                id2subject_map,             
                                                                  input_ids,
                                                                  tokenizer,
                                                                  batch_origin_info,
                                                                  offset_mapping,
                                                                  all_known_subjects
                                                                  )         

            all_subjects.append(batch_subjects)
            all_subject_labels.append(batch_subject_labels)    
            # 需要判断 batch_subjects 是空的情况，最好能够和普通subjects 一样处理        
            if(len(batch_subjects[0]) == 0):
                #print("----- 未预测到subject ----------")
                invalid_num+=1
                continue

            logger.info(f"当前文本是：{batch_origin_info}")
            logger.info(f"损失是：{cur_loss}")
        
        # 写入到文件中(w)
        #visualize_subject(pred_file_path, all_subjects)
        visualize_subject_with_label(pred_file_path, all_subjects, all_subject_labels)
        avg_loss = total_loss / len(dev_data_loader)
        logger.info(f"平均损失是：{avg_loss}")
        logger.info(f"未预测到的subject 数目是：{invalid_num}")


def do_train():
    # ========================== subtask 1. 预测subject ==========================
    # 这一部分我用一个 NER 任务来做，但是原任务用的是 start + end 的方式，原理是一样的
    # ========================== =================== =============================
    bert_name_or_path = "/home/lawson/pretrain/bert-base-chinese"
    roberta_name_or_path = "/pretrains/pt/chinese_RoBERTa-wwm-ext_pytorch"
    model_subject = SubjectModel(bert_name_or_path,768,out_fea=subject_class_num) 
    if (args.init_checkpoint != None): # 加载初始模型
        model_subject.load_state_dict(t.load(args.init_checkpoint))
    model_subject = model_subject.cuda()    
    # crf = CRF(num_tags = subject_class_num,batch_first=True)
    # crf = crf.cuda()
    #print(crf.transitions)
    # 这里将DistributedBatchSample(paddle) 修改成了 DistributedSample(torch)    
    # 如果使用 DistributedSampler 那么应该就是一个多进程加载数据
    # train_batch_sampler = DistributedSampler(
    #     train_dataset,
    #     shuffle=True,
    #     drop_last=True 
    #     )
    
    # Loads dataset.
    train_dataset = TrainSubjectDataset.from_file(
        args.train_data_path,
        tokenizer,
        args.max_seq_length,
        True
        )
    # crf.transitions
    train_data_loader = DataLoader(        
        dataset=train_dataset,
        #batch_sampler=train_batch_sampler,
        batch_size=args.batch_size,
        collate_fn=collator, # 重写一个 collator
        )

     # Loads dataset.
    # 放在外面是为了避免每次 evaluate 的时候都加载一遍
    # dev 数据集也是用 TrainSubjectDataset 的原因是：想计算loss
    dev_dataset = TrainSubjectDataset.from_file(        
        args.dev_data_path,
        tokenizer,
        args.max_seq_length,
        True
        )

    dev_data_loader = DataLoader(        
        dataset=dev_dataset,        
        batch_size=args.batch_size,
        collate_fn=collator, # 重写一个 collator
        )

    # 这里为什么只对一部分的参数做这个decay 操作？ 这个decay 操作有什么作用？
    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model_subject.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    
    # 需要合并所有模型的参数    
    optimizer = t.optim.Adam(
        [
        {'params':model_subject.parameters(),'lr':2e-5},
        #{'params':crf.parameters(),'lr':0.1},
        ],
        )
    
    # Defines learning rate strategy.
    steps_by_epoch = len(train_data_loader)
    num_training_steps = steps_by_epoch * args.num_train_epochs    
    lr_scheduler = ReduceLROnPlateau(optimizer=optimizer,
                                     mode='min')
    
    # 打印本次的配置
    logger.info("the paramers in this model are:")
    for k,v in (vars(args).items()):
        logger.info(f"{k,v}")


    # Starts training.
    global_step = 0
    logging_steps = 50
    save_steps = 5000
    max_f1 = 0 # 最佳f1    
    for epoch in tqdm(range(args.num_train_epochs)):
        #print(crf.transitions)
        logger.info(f"\n=====start training of {epoch} epochs =====")        
        # 设置为训练模式
        model_subject.train() # 预测subject
        step = 0        
        vis_loss = 0 # 输出到visdom 的loss
        for batch in tqdm(train_data_loader):            
            step += 1
            input_ids,token_type_ids,attention_mask,batch_origin_info, labels,offset_mappings = batch
            # labels size = [batch_size,max_seq_length]
            logits_1 = model_subject(input_ids=input_ids,
                                   token_type_ids=token_type_ids,
                                   attention_mask=attention_mask
                                   )
            
            # batch_size = logits_1.size(0)
            # max_seq_length = logits_1.size(1)
            # label_num = logits_1.size(2)
            # logits_1 = logits_1.view(max_seq_length,batch_size,label_num) # reshape 至可以让crf处理
            
            # labels = labels.view(max_seq_length,batch_size)
            # attention_mask = attention_mask.view(max_seq_length,batch_size)            
            # 添加crf
            #loss = -crf(logits_1, labels, mask = attention_mask.byte(), reduction = 'mean')

            #logits size [batch_size,max_seq_len,class_num]
            logits_1 = logits_1.view(-1,subject_class_num)
            labels = labels.view(-1)
            loss = criterion(logits_1, labels)
            
            loss.backward()
            optimizer.step()
            #lr_scheduler.step()
            optimizer.zero_grad()
            loss_item = loss.item()
            vis_loss += loss_item
            
            logger.info(f"epoch:{epoch}/{args.num_train_epochs},  steps:{step}/{steps_by_epoch},   loss:{loss_item}")
            if loss_item > 0.1 :
                logger.info(f"{batch_origin_info}")
            # 打日志
            if global_step % logging_steps == 0 and global_step:                
                vis.line([vis_loss], [global_step], win=win, update="append")
                vis_loss = 0            
            global_step += 1

            if global_step % save_steps == 0 and global_step:
                save_model_path = os.path.join(args.output_dir,"model_subject_%d_bert.pdparams" % (global_step+53530))
                logger.info("saving checkpoing model_subject_%d_bert.pdparams to %s " %
                        (global_step, args.output_dir))
                t.save(model_subject.state_dict(),save_model_path)

        
        # 使用dev 数据集评测模型效果
        pred_file_path = f"/home/lawson/program/DuIE_py/data/predict/dev_data_subject_predict_model_subject_{global_step+53530}_bert.txt"
        evaluate(model_subject,dev_data_loader,criterion,pred_file_path,crf=None,all_known_subjects=None)
        recall,precision,f1 = cal_subject_metric(dev_data_file_path = args.dev_data_path, pred_file_path=pred_file_path)
        if f1 > max_f1 :
            # 选择最佳f1 值保存模型
            logger.info(f"saving checkpoing model_subject_{global_step}.pdparams to {args.output_dir}")
            cur_model_subject_name = os.path.join(args.output_dir,"model_subject_%d_bert_f1=%f.pdparams" % (global_step+53530,f1))
            #cur_model_crf_name = os.path.join(args.output_dir,"crf_%d_bert.pdparams" % (global_step))
            t.save(model_subject.state_dict(),cur_model_subject_name)
            #t.save(crf.state_dict(),cur_model_crf_name)
            max_f1 = f1
    
        logger.info(f"recall = {recall}, precision = {precision}, f1 = {f1}")        
        
    logger.info("\n=====training complete=====")


if __name__ == "__main__":
    set_random_seed(args.seed)
    # 指定GPU设备    
    device = t.device("cuda" if t.cuda.is_available() and not args.n_gpu else "cpu")    

    if args.do_train:
        do_train()
    if args.do_eval:        
        roberta_name_or_path = "/pretrains/pt/chinese_RoBERTa-wwm-ext_pytorch"
        bert_name_or_path = "/home/lawson/pretrain/bert-base-chinese"
        model_subject = SubjectModel(bert_name_or_path,768,out_fea=subject_class_num)
        if (args.init_checkpoint != None): # 加载初始模型
            model_subject.load_state_dict(t.load(args.init_checkpoint))
        model_subject = model_subject.cuda()
        
        collator = TrainSubjectDataCollator()
        # Loads dataset.
        # 放在外面是为了避免每次 evaluate 的时候都加载一遍
        # dev 数据集也是用 TrainSubjectDataset 的原因是：想计算loss
        dev_dataset = TrainSubjectDataset.from_file(        
            args.dev_data_path,
            tokenizer,
            args.max_seq_length,
            True
            )

        dev_data_loader = DataLoader(
            dataset=dev_dataset,
            batch_size=args.batch_size,
            collate_fn=collator, # 重写一个 collator
            )
        
        # 找出训练数据集中已知的所有subjects 
        all_known_subjects = get_all_subjects(train_data_path=args.dev_data_path)
        temp1 = (args.dev_data_path).split("/")[-1].split(".")[0]
        temp2 = (args.init_checkpoint).split("/")[-1]
        pred_file_path = f"/home/lawson/program/DuIE_py/data/predict/{temp1}_predict_subject_{temp2}_7_3.txt"
        if os.path.exists(pred_file_path):
            os.remove(pred_file_path)
        evaluate(model_subject,dev_data_loader,criterion,pred_file_path,crf=None,all_known_subjects=all_known_subjects)
        recall,precision,f1 = cal_subject_metric(args.dev_data_path,pred_file_path=pred_file_path)