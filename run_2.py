"""
训练 object的模型

20210411
01.添加预测为空类，即有subject也可能预测不到 object   => 主要对应人名的系列
02.在训练object的模型时，使用的也是TrainSubjectDataset，因为需要从一个基础的数据集得到 subject，从而进行拼接生成object的训练数据集
"""
from metric import cal_object_metric
from visdom import Visdom 
from transformers import BertTokenizerFast
from models import ObjectModel
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

from data_loader import  TrainSubjectDataset,TrainSubjectDataCollator
from utils import decode_subject,decode_object, decoding, find_entity, get_precision_recall_f1, write_prediction_results

from data_loader import from_dict2object,from_dict2_relation

parser = argparse.ArgumentParser()
parser.add_argument("--do_train", action='store_true', default=False, help="do train")
parser.add_argument("--do_predict", action='store_true', default=False, help="do predict")
parser.add_argument("--init_checkpoint", default=None, type=str, required=False, help="Path to initialize params from")
parser.add_argument("--data_path", default="./data", type=str, required=False, help="Path to data.")
parser.add_argument("--train_data_path", default="./data", type=str, required=False, help="Path to data.")
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


# Reads object_map.
object_map_path = os.path.join(args.data_path, "object2id.json")
if not (os.path.exists(object_map_path) and os.path.isfile(object_map_path)):
    sys.exit("{} dose not exists or is not a file.".format(object_map_path))
with open(object_map_path, 'r', encoding='utf8') as fp:
    object_map = json.load(fp)

object_class_num = len(object_map.keys())  # 得出object 的class num    

# Reads object_map.
id2object_map_path = os.path.join(args.data_path, "id2object.json")
if not (os.path.exists(id2object_map_path) and os.path.isfile(id2object_map_path)):
    sys.exit("{} dose not exists or is not a file.".format(id2object_map_path))
with open(id2object_map_path, 'r', encoding='utf8') as fp:
    id2object_map = json.load(fp)


name_or_path = "/pretrains/pt/chinese_RoBERTa-wwm-ext_pytorch"
model_object = ObjectModel(name_or_path,768,object_class_num)
if args.init_checkpoint is not None and os.path.exists(args.init_checkpoint):
    model_object.load_state_dict(t.load(args.init_checkpoint))
model_object = model_object.cuda()
criterion = nn.CrossEntropyLoss() # 使用交叉熵计算损失
tokenizer = BertTokenizerFast.from_pretrained("/home/lawson/pretrain/bert-base-chinese")
collator = TrainSubjectDataCollator()

# 在object数据集中使用TrainSubjectDataset 是没有问题的    
dev_dataset = TrainSubjectDataset.from_file(args.dev_data_path,
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


viz_object = Visdom()
win = "train_object_loss"


def set_random_seed(seed):
    """sets random seed"""
    random.seed(seed)
    np.random.seed(seed)
    #t.seed(seed)  # 为什么torch 也要设置这个seed ？

import time
curTime = time.strftime("%m%d_%H%M%S", time.localtime())
log_name = "object_" + curTime + '_.log'
logging.basicConfig(format='%(asctime)s - %(levelname)s -%(name)s - %(message)s',
                    datefmt='%m/%d%/%Y %H:%M:%S',
                    level=logging.INFO,
                    filemode='w',
                    filename='/home/lawson/program/DuIE_py/log/' + log_name
                    )
logger = logging.getLogger("object")



"""
可视化object的预测结果
"""
def visualize_object(pred_file_path, all_objects, all_object_labels):
    with open(pred_file_path,'w') as f:
        for item in zip(all_objects,all_object_labels):
            objects,labels = item
            f.write(str(objects) +"\n")
            f.write(str(labels) + "\n")
            f.write("\n")



def evaluate(model_object,dev_data_loader,criterion,pred_file_path):
    # Does predictions.
    logger.info("\n====================start  evaluating ====================")   
    tokenizer = BertTokenizerFast.from_pretrained("/pretrains/pt/chinese_RoBERTa-wwm-ext_pytorch")    
    # 将object的预测结果写到文件中
    if os.path.exists(pred_file_path):# 因为下面是追加写入到文件中，所以
        os.remove(pred_file_path)
    total_loss = 0 # 总损失
    invalid_num = 0 # 预测失败的个数
    all_objects = []
    all_object_labels = []
    with t.no_grad():
        for batch in tqdm(dev_data_loader):            
            input_ids,token_type_ids,attention_mask, batch_origin_info,labels,batch_offset_mapping = batch

            object_input_ids, object_token_type_ids,object_attention_mask, object_labels,object_origin_info,batch_object_offset_mapping= from_dict2object(batch_subjects=None, batch_origin_dict=batch_origin_info,tokenizer=tokenizer,max_length=args.max_seq_length,pad_to_max_length = True)

            object_input_ids = t.tensor(object_input_ids).cuda()
            object_token_type_ids = t.tensor(object_token_type_ids).cuda()
            object_attention_mask = t.tensor(object_attention_mask).cuda()            
            object_labels = t.tensor(object_labels).cuda()
            logits_2 = model_object(input_ids = object_input_ids,
                                    token_type_ids=object_token_type_ids,
                                    attention_mask=object_attention_mask
                                    ) # size [batch_size,max_seq_len,object_class_num]
            logits = logits_2 #备份用于后面decode 
            logits_2 = logits_2.view(-1,object_class_num)
            object_labels = object_labels.view(-1)  
            cur_loss = criterion(logits_2,object_labels)


            total_loss += cur_loss
            # 得到预测到的 subject            
            batch_objects, batch_object_labels = decode_object(logits=logits,
                                                               id2object_map=id2object_map,
                                                               batch_object_input_ids=object_input_ids,
                                                               tokenizer=tokenizer,
                                                               batch_object_origin_info=object_origin_info,
                                                               batch_object_offset_mapping=batch_object_offset_mapping
                                                               )
            
            all_objects.extend(batch_objects)
            all_object_labels.extend(batch_object_labels)
    
        # 写入到文件中(w)
        visualize_object(pred_file_path, all_objects, all_object_labels)
        recall,precision,f1 = cal_object_metric(all_objects,args.dev_data_path)
        logger.info(f"recall={recall},precision={precision},f1={f1}")
        print(f"recall={recall},precision={precision},f1={f1}")
        avg_loss = total_loss / len(dev_data_loader)
        logger.info(f"平均损失是：{avg_loss}")
        logger.info(f"未预测到的object 数目是：{invalid_num}")



def do_train():
    # ========================== subtask 1. 预测 object ==========================    
    if args.init_checkpoint is not None and os.path.exists(args.init_checkpoint):
        logger.info(f"加载模型:{args.init_checkpoint}")
        model_object.load_state_dict(t.load(args.init_checkpoint))
    
    
    # Loads dataset.
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
    train_data_loader = DataLoader(        
        dataset=train_dataset,
        #batch_sampler=train_batch_sampler,
        batch_size=args.batch_size,
        collate_fn=collator, # 重写一个 collator
        shuffle=True
        )

    # 需要合并所有模型的参数    
    optimizer = t.optim.AdamW(
        [{'params':model_object.parameters(),'lr':2e-5},        
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
    logging_steps = 500
    save_steps = 5000
    tic_train = time.time()
    for epoch in tqdm(range(args.num_train_epochs)):
        logger.info("\n=====start training of %d epochs=====" % epoch)
        tic_epoch = time.time()
        # 设置为训练模式        
        model_object.train() # 根据subject 预测object     
        step = 1   
        for batch in tqdm(train_data_loader):
            input_ids,token_type_ids,attention_mask, batch_origin_info,labels, batch_offset_mapping = batch

            # ====== 根据origin_info 得到 subtask 2 的训练数据 ==========
            # 这里的object_input_ids 的size 不再是args.batch_size ，可能比这个稍大

            object_input_ids, object_token_type_ids,object_attention_mask, object_labels,object_origin_info,object_offset_mapping= from_dict2object(batch_subjects=None, batch_origin_dict=batch_origin_info,tokenizer=tokenizer,max_length=args.max_seq_length,pad_to_max_length = True)

            object_input_ids = t.tensor(object_input_ids).cuda()
            object_token_type_ids = t.tensor(object_token_type_ids).cuda()
            object_attention_mask = t.tensor(object_attention_mask).cuda()            
            object_labels = t.tensor(object_labels).cuda()
            logits_2 = model_object(input_ids = object_input_ids,
                                    token_type_ids=object_token_type_ids,
                                    attention_mask=object_attention_mask
                                    ) # size [batch_size,max_seq_len,object_class_num]
            logits_2 = logits_2.view(-1,object_class_num) 
            object_labels = object_labels.view(-1)  
            loss = criterion(logits_2,object_labels)
            
            loss.backward()
            optimizer.step()
            #lr_scheduler.step()
            optimizer.zero_grad()
            loss_item = loss.item()

            if global_step % logging_steps == 0 :
                logger.info(
                    "epoch: %d / %d, steps: %d / %d, loss: %f , speed: %.2f step/s"
                    % (epoch, args.num_train_epochs, step, steps_by_epoch,
                       loss_item,
                       logging_steps / (time.time() - tic_train))
                       )
                tic_train = time.time()

            if global_step % save_steps == 0:
                logger.info("saving checkpoing model_object_%d_bert.pdparams to %s " %
                        (global_step, args.output_dir))
                t.save(model_object.state_dict(),
                            os.path.join(args.output_dir,"model_object_%d_bert.pdparams" % global_step))
            step+=1 
            global_step += 1 
            viz_object.line([loss_item], [global_step], win=win, update="append")
        t.save(model_object.state_dict(),os.path.join(args.output_dir,
                            "model_object_%d_bert.pdparams" % global_step))
        logger.info("\n=====training complete=====")


if __name__ == "__main__":
    set_random_seed(args.seed)
    # 指定GPU设备    
    device = t.device("cuda" if t.cuda.is_available() and not args.n_gpu else "cpu")

    if args.do_train:
        do_train()
        pred_file_path = "/home/lawson/program/DuIE_py/data/object_predict.txt"
        #evaluate(model_object,dev_data_loader,criterion,pred_file_path)