'''
Author: LawsonAbs
Date: 2021-04-06 17:53:10
LastEditTime: 2021-04-06 21:44:51
FilePath: /duie_py/models.py
'''
import torch as t
import torch.nn as nn
from torch.nn.parameter import Parameter
from transformers import BertConfig,BertModel,BertForSequenceClassification
from torchcrf import CRF
from transformers.utils.dummy_tokenizers_objects import BertTokenizerFast

class SubjectModel(nn.Module):
    def __init__(self,name_or_path,in_fea,out_fea):
        super(SubjectModel,self).__init__()
        self.model = BertModel.from_pretrained(name_or_path)
        self.linear = nn.Linear(in_fea,out_fea)        

    def forward(self,input_ids,token_type_ids,attention_mask):
        out = self.model(input_ids=input_ids,
                         token_type_ids=token_type_ids,
                         attention_mask=attention_mask
                         )
        last_layer = out.last_hidden_state
        # size = [batch_size,max_seq_length,768]
        out = self.linear(last_layer)
        return out
        # size = [batch_size,max_seq_length,out_fea]

'''
功能：预测Object 的模型
'''
class ObjectModel(nn.Module):
    def __init__(self,name_or_path,in_fea,out_fea):
        super(ObjectModel,self).__init__()
        self.model = BertModel.from_pretrained(name_or_path)
        self.linear = nn.Linear(in_fea,out_fea)

    def forward(self,input_ids,token_type_ids,attention_mask):
        out = self.model(input_ids=input_ids,
                         token_type_ids=token_type_ids,
                         attention_mask=attention_mask
                         )
        last_layer = out.last_hidden_state
        #last_layer,pool_layer = out
        # size = [batch_size,max_seq_length,768]
        out = self.linear(last_layer)                   
        return out



class RelationModel(nn.Module):
    def __init__(self,name_or_path,relation_class_num):
        super(RelationModel,self).__init__()     
        # 这里重新配置了一个config，是为了使用 BertForSequenceClassfication 进行文本分类
        self.config = BertConfig.from_pretrained(name_or_path,num_labels = relation_class_num)
                
        # 可以不用传label，直接对应输出 logits
        #self.model = BertForSequenceClassification.from_pretrained(name_or_path, config=self.config)
        self.model = BertModel.from_pretrained(name_or_path, config=self.config)
        #self.my_position_embedding = t.randn((256,768),requires_grad=True,device='cuda') # 随机初始化一个位置向量，用于训练前后的关系
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768,relation_class_num)
        
        self.my_position_embedding =  nn.Embedding(256,768)
        self.tokenizer = BertTokenizerFast.from_pretrained(name_or_path)
        # self.register_parameter(name='my_position_embedding',param=Parameter(self.my_position_embedding))

    def forward(self,input_ids,token_type_ids,attention_mask,labels,indexs):
        outputs = self.model(input_ids=input_ids,
                         token_type_ids=token_type_ids,
                         attention_mask=attention_mask,    
                         #labels = labels
                         #output_hidden_states = True
                         )
        #cls_emb = out.last_hidden_state[:,0,:]        
        #cls_emb = out.hidden_states[12][:,0,:]
        
        
        cur_position_emb = self.my_position_embedding(indexs)            
        temp = outputs[1]
        pooled_output = temp + cur_position_emb
        logits = self.classifier(pooled_output)
        return logits
        
    
        