import torch as t
import torch.nn as nn
from transformers import BertConfig,BertModel,BertForSequenceClassification

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
        # size = [batch_size,max_seq_length,768]
        out = self.linear(last_layer)                   
        return out

class RelationModel(nn.Module):
    def __init__(self,name_or_path,in_fea,out_fea):
        super(RelationModel,self).__init__()     
        # 这里重新配置了一个config，是为了使用 BertForSequenceClassfication 进行文本分类
        self.config = BertConfig.from_pretrained(name_or_path,num_labels = 48)        
        self.model = BertForSequenceClassification.from_pretrained(name_or_path, config=self.config)
        self.linear = nn.Linear(in_fea,out_fea)

    def forward(self,input_ids,token_type_ids,attention_mask,labels):
        out = self.model(input_ids=input_ids,
                         token_type_ids=token_type_ids,
                         attention_mask=attention_mask,
                         labels = labels
                         )        
        return out