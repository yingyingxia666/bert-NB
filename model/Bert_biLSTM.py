import torch
from torch import nn
from transformers import BertModel,BertConfig,BertTokenizer


class Bert_biLSTM(nn.Module):

    def __init__(self, pretrained_path, input_size=768, hidden_size=768//2, num_class=2):
        super(Bert_biLSTM,self).__init__()
        self.config = BertConfig.from_pretrained(pretrained_path)
        self.bert = BertModel.from_pretrained(pretrained_path,config=self.config)
        # for param in self.bert.parameters():
        #     param.requires_grad = False
        self.rnn = nn.LSTM(bidirectional=True, num_layers=2, input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size*2, num_class)
    def forward(self, x):
        output= self.bert(x).pooler_output
        output, _ = self.rnn(output)
        output = self.fc(output)
        return output


if __name__=='__main__':
    pretrained_path = 'bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(pretrained_path)
    # encode_plus 返回一个字典
    batch = tokenizer.encode_plus(['这是一个晴朗的早晨'],return_tensors = 'pt') 
    x=batch['input_ids']
    model=Bert_biLSTM(pretrained_path,2)
    y_test=model(x)
    print(y_test)
