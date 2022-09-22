from re import X
import torch
from torch import nn
from transformers import BertModel,BertConfig,BertTokenizer


class Bert_biLSTM(nn.Module):

    def __init__(self, pretrained_path, num_class):
        super(Bert_biLSTM,self).__init__()
        self.config = BertConfig.from_pretrained(pretrained_path)
        self.bert = BertModel.from_pretrained(pretrained_path,config=self.config)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.rnn = nn.LSTM(bidirectional=True, num_layers=2, input_size=768, hidden_size=768//2, batch_first=True)
        self.fc = nn.Linear(768, num_class)
    def forward(self,x):
        x = torch.tensor(x)
        # self.bert.train()
        output= self.bert(x).pooler_output
        output, _ = self.rnn(output)
        logits = self.fc(output) 
        y_hat = logits.argmax(-1)
        return y_hat


        # self.bert.eval()
        # with torch.no_grad():
        #         x = torch.tensor(x)
        #         embedding = self.bert(x)
        # output = embedding.pooler_output
        # output = self.fc(output)
        # return output


if __name__=='__main__':
    pretrained_path = 'bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(pretrained_path)
    # encode_plus 返回一个字典
    batch = tokenizer.encode_plus(['这是一个晴朗的早晨']) 
    x=[batch['input_ids']]
    # import pdb
    # pdb.set_trace()
    model=Bert_biLSTM(pretrained_path,2)
    # model.train()
    y_test=model(x)
    y=1
    print(y_test)
    ##
    # import torch
    # from torch import nn
    # from transformers import BertModel,BertConfig,BertTokenizer

    # # 预训练模型存储位置
    # pretrained_path = 'bert-base-chinese'
    # config = BertConfig.from_pretrained(pretrained_path)
    # tokenizer = BertTokenizer.from_pretrained(pretrained_path)
    # model = BertModel.from_pretrained(pretrained_path,config=config)

    # batch = tokenizer.encode_plus('这是一个晴朗的早晨') # encode_plus 返回一个字典
    # input_ids = torch.tensor([batch['input_ids']])
    # token_type_ids = torch.tensor([batch['token_type_ids']])
    # attention_mask = torch.tensor([batch['attention_mask']])
    # import pdb
    # pdb.set_trace()
    # embedding = model(input_ids,token_type_ids=token_type_ids)
