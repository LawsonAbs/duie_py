import torch as t
import torch.nn as nn
from transformers import BertModel

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