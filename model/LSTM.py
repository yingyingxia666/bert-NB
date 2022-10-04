import torch
from torch import nn
from transformers import BertModel,BertConfig,BertTokenizer


class LSTM(nn.Module):

    def __init__(self, input_size=768, hidden_size=768//2, num_layers=2, num_class=2):
        super(LSTM,self).__init__()
        self.rnn = nn.LSTM(bidirectional=False, num_layers=2, input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_class)
    def forward(self, x):
        output, _ = self.rnn(x)
        output = self.fc(output) 
        return output

