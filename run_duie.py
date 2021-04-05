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
from transformers.utils.dummy_pt_objects import BertModel

from data_loader import DuIEDataset, DataCollator, ObjectDataset,SubjectDataset,SubjectDataCollator
from data_loader import PredictSubjectDataset,PredictSubjectDataCollator
from utils import decode_subject,decode_object, decoding, find_entity, get_precision_recall_f1, write_prediction_results

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


class BCELossForDuIE(nn.Module):
    def __init__(self, ):
        super(BCELossForDuIE, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, labels, mask):
        loss = self.criterion(logits, labels)
        # TODO : 研究cast 的用法 
        mask = t.cast(mask, 'float32')  
        loss = loss * mask.unsqueeze(-1)
        loss = t.sum(loss.mean(axis=2), axis=1) / t.sum(mask, axis=1)
        loss = loss.mean()
        return loss


def set_random_seed(seed):
    """sets random seed"""
    random.seed(seed)
    np.random.seed(seed)
    #t.seed(seed)  # 为什么torch 也要设置这个seed ？

logging.basicConfig(format='%(asctime)s - %(levelname)s -%(name)s - %(message)s',
                    datefmt='%m/%d%/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger("duie")


@t.no_grad()
def evaluate(model_subject, criterion, data_loader, file_path, mode):
    """
    mode eval:
    eval on development set and compute P/R/F1, called between training.
    mode predict:
    eval on development / test set, then write predictions to \
        predict_test.json and predict_test.json.zip \
        under args.data_path dir for later submission or evaluation.
    """
    model_subject.eval()
    probs_all = None
    seq_len_all = None
    tok_to_orig_start_index_all = None
    tok_to_orig_end_index_all = None
    loss_all = 0
    eval_steps = 0
    for batch in tqdm(data_loader, total=len(data_loader)):
        eval_steps += 1
        input_ids,token_type_ids,attention_mask, seq_lens, labels = batch
        logits = model_subject(input_ids=input_ids)
        #mask = (input_ids != 0).logical_and((input_ids != 1)).logical_and((input_ids != 2)) # 作用是什么？
        loss = criterion(logits, labels )
        loss_all += loss.item()
        probs = F.sigmoid(logits)
        if probs_all is None:
            probs_all = probs.numpy()
            seq_len_all = seq_len.numpy()            
        else:
            probs_all = np.append(probs_all, probs.numpy(), axis=0)
            seq_len_all = np.append(seq_len_all, seq_len.numpy(), axis=0)                        
    loss_avg = loss_all / eval_steps
    print("eval loss: %f" % (loss_avg))

    id2spo_path = os.path.join(os.path.dirname(file_path), "id2spo.json")
    with open(id2spo_path, 'r', encoding='utf8') as fp:
        id2spo = json.load(fp)
    formatted_outputs = decoding(file_path, id2spo, probs_all, seq_len_all,
                                 tok_to_orig_start_index_all,
                                 tok_to_orig_end_index_all)
    
    
    if mode == "predict":
        predict_file_path = os.path.join(args.data_path, 'predictions.json')
    else:
        predict_file_path = os.path.join(args.data_path, 'predict_eval.json')    

    predict_zipfile_path = write_prediction_results(formatted_outputs,
                                                    predict_file_path)
    
    # 在 eval 中添加评测可视化部分，即将评测结果也写到文件中
    if mode == "eval":
        precision, recall, f1 = get_precision_recall_f1(file_path,
                                                        predict_zipfile_path)
        # 不要删除这个文件
        #os.system('rm {} {}'.format(predict_file_path, predict_zipfile_path))
        return precision, recall, f1
    elif mode != "predict":
        raise Exception("wrong mode for eval func")


def do_train():
    # ========================== subtask 1. 预测subject ==========================
    # 这一部分我用一个 NER 任务来做，但是原任务用的是 start + end 的方式，原理是一样的
    # ========================== =================== ==========================    
    name_or_path = "/home/lawson/pretrain/bert-base-chinese"
    model_subject = SubjectModel(name_or_path,768,out_fea=subject_class_num)    
    model_subject = model_subject.cuda()
    model_object = ObjectModel(name_or_path,768,object_class_num)        
    model_object = model_object.cuda()

    model_relation = RelationModel(name_or_path,relation_class_num)
    model_relation = model_relation.cuda()
    tokenizer = BertTokenizer.from_pretrained("/home/lawson/pretrain/bert-base-chinese")
    criterion = nn.CrossEntropyLoss() # 使用交叉熵计算损失

    # Loads dataset.
    train_dataset = SubjectDataset.from_file(
        os.path.join(args.data_path, 'train_data.json'),
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
    collator = SubjectDataCollator()
    train_data_loader = DataLoader(        
        dataset=train_dataset,
        #batch_sampler=train_batch_sampler,
        batch_size=args.batch_size,
        collate_fn=collator, # 重写一个 collator
        )
    
    dev_file_path = os.path.join(args.data_path, 'dev_data_200.json')
    dev_dataset = SubjectDataset.from_file(dev_file_path,
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
    
    # 这里为什么只对一部分的参数做这个decay 操作？ 这个decay 操作有什么作用？
    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model_subject.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    
    # 需要合并所有模型的参数    
    optimizer = t.optim.AdamW(
        [{'params':model_subject.parameters(),'lr':2e-5},
        {'params':model_object.parameters(),'lr':2e-5},
        {'params':model_relation.parameters(),'lr':2e-5}        
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
    logging_steps = 50
    save_steps = 10000
    tic_train = time.time()
    for epoch in range(args.num_train_epochs):
        print("\n=====start training of %d epochs=====" % epoch)
        tic_epoch = time.time()
        # 设置为训练模式
        model_subject.train() # 预测subject
        model_object.train() # 根据subject 预测object
        model_relation.train() # 根据subject+object 预测 relation
        for step, batch in enumerate(train_data_loader):
            input_ids,token_type_ids,attention_mask, labels,origin_info = batch
            # labels size = [batch_size,max_seq_length]
            logits_1 = model_subject(input_ids=input_ids,
                                   token_type_ids=token_type_ids,
                                   attention_mask=attention_mask
                                   )
            #logits size [batch_size,max_seq_len,class_num]
            logits_1 = logits_1.view(-1,subject_class_num) 
            labels = labels.view(-1)  
            loss_1 = criterion(logits_1, labels)
            
            # ====== 根据origin_info 得到 subtask 2 的训练数据 ==========
            # 这里的object_input_ids 的size 不再是args.batch_size ，可能比这个稍大
            object_input_ids, object_token_type_ids,object_attention_mask, object_labels = from_dict(subjects=None, batch_origin_dict=origin_info,tokenizer=tokenizer,max_length=args.max_seq_length,pad_to_max_length = True)
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
            loss_2 = criterion(logits_2,object_labels)
            
            # ====== 根据origin_info 得到 subtask 3 的训练数据 ==========
            # 根据 subject + object 预测 关系
            relation_input_ids, relation_token_type_ids, relation_attention_mask, relation_labels = from_dict2_relation(subjects=None,objects=None, batch_origin_info= origin_info ,tokenizer=tokenizer,max_length=args.max_seq_length)
            
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
            
            loss_3 = out.loss
            loss = loss_1 + loss_2 + loss_3
            loss.backward()
            optimizer.step()
            #lr_scheduler.step()
            optimizer.zero_grad()
            loss_item = loss.item()

            if global_step % logging_steps == 0 :
                print(
                    "epoch: %d / %d, steps: %d / %d, loss: %f # loss_1:%f #loss_2:%f,#loss_3:%f, speed: %.2f step/s"
                    % (epoch, args.num_train_epochs, step, steps_by_epoch,
                       loss_item, loss_1.item(),loss_2.item(),loss_3.item(),
                       logging_steps / (time.time() - tic_train))
                       )
                tic_train = time.time()
            '''
            if global_step % save_steps == 0 and global_step != 0 :
                print("\n=====start evaluating ckpt of %d steps=====" %
                      global_step)
                precision, recall, f1 = evaluate(
                    model_subject, criterion, dev_data_loader, dev_file_path, "eval")
                print("precision: %.2f\t recall: %.2f\t f1: %.2f\t" %
                      (100 * precision, 100 * recall, 100 * f1))
                if (not args.n_gpu > 1) or t.distributed.get_rank() == 0:
                    print("saving checkpoing model_subject_%d.pdparams to %s " %
                          (global_step, args.output_dir))
                    t.save(model_subject.state_dict(),
                                os.path.join(args.output_dir,
                                             "model_subject_%d.pdparams" % global_step))
                model_subject.train()  # back to train mode
            '''     
            global_step += 1
        # tic_epoch = time.time() - tic_epoch
        # print("epoch time footprint: %d hour %d min %d sec" %
        #       (tic_epoch // 3600, (tic_epoch % 3600) // 60, tic_epoch % 60))

        '''
        # Does final evaluation.    
        print("\n=====start evaluating last ckpt of %d steps=====" %
                global_step)
        precision, recall, f1 = evaluate(model_subject, 
                                        criterion,
                                        dev_data_loader,
                                        dev_file_path,
                                        "eval")
        print("precision: %.2f\t recall: %.2f\t f1: %.2f\t" %
                (100 * precision, 100 * recall, 100 * f1))
        '''
        t.save(model_subject.state_dict(),
            os.path.join(args.output_dir,
                            "model_subject_%d.pdparams" % global_step)
            )
        t.save(model_object.state_dict(),os.path.join(args.output_dir,
                            "model_object_%d.pdparams" % global_step))
        t.save(model_relation.state_dict(),os.path.join(args.output_dir,
                            "model_relation_%d.pdparams" % global_step))
        print("\n=====training complete=====")



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
        os.path.join(args.data_path, 'dev_data_200.json'),
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
    for step, batch in enumerate(dev_data_loader):
        input_ids,token_type_ids,attention_mask, origin_info = batch
        # labels size = [batch_size,max_seq_length]
        logits_1 = model_subject(input_ids=input_ids,
                                token_type_ids=token_type_ids,
                                attention_mask=attention_mask
                                )
        #logits size [batch_size,max_seq_len,class_num]  
        # 得到预测到的 subject
        subjects,subject_labels = decode_subject(logits_1,id2subject_map,input_ids,tokenizer) 
        # ====== 根据origin_info 得到 subtask 2 的训练数据 ==========
        # 这里的object_input_ids 的 size 不再是args.batch_size ，可能比这个稍大
        object_input_ids, object_token_type_ids,object_attention_mask, object_labels = from_dict(subjects,
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
        objects,object_label = decode_object(logits_2,id2object_map,tokenizer,input_ids)
        
        # ====== 根据 subject + object 得到 subtask 3 的训练数据 ==========        
        relation_input_ids, relation_token_type_ids, relation_attention_mask, relation_labels = from_dict2_relation(subjects,objects,origin_info,tokenizer,args.max_seq_length,)
        
        relation_input_ids = t.tensor(relation_input_ids).cuda()
        relation_token_type_ids = t.tensor(relation_token_type_ids).cuda()
        relation_attention_mask = t.tensor(relation_attention_mask).cuda()        

        # 这个模型直接得到loss
        out = model_relation(input_ids=relation_input_ids,
                                token_type_ids=relation_token_type_ids,
                                attention_mask=relation_attention_mask,
                                labels = relation_labels
                                )
        
        
        # 输出最后的结果
    print("=====predicting complete=====")


if __name__ == "__main__":
    set_random_seed(args.seed)
    # 指定GPU设备    
    device = t.device("cuda" if t.cuda.is_available() and not args.n_gpu else "cpu")    

    if args.do_train:        
        do_train()
    if args.do_predict:
        model_subject_path = "./checkpoints/model_subject_30000.pdparams"
        model_object_path = "./checkpoints/model_object_5000.pdparams"
        model_relation_path  = "./checkpoints/model_relation_5000.pdparams"        
        do_predict(model_subject_path, model_object_path,model_relation_path)
