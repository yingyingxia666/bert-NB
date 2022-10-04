import torch
from torch import nn
import numpy as np
from transformers import (
    BertTokenizer,
)
import jieba
from torch.utils.data import DataLoader
from config import train_config
from tqdm import tqdm
from model.LSTM import LSTM
from model.Bert_biLSTM import Bert_biLSTM
from dataset import dataset
from sklearn import metrics

def train(train_config):
    #device设置
    if train_config['gpu']:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print(device)
    #优化器、损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr = train_config['lr'])
    loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn = loss_fn.to(device)
    #数据集构建
    traindataset = dataset(train['data_dir']+'/train.csv')
    testdataset = dataset(train['data_dir']+'/test.csv')
    valdataset = dataset(train['data_dir']+'/val.csv')
    train_dataloader = DataLoader(traindataset, batch_size=train_config['batch_size'], shuffle=True)
    test_dataloader = DataLoader(testdataset, batch_size=train_config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(valdataset, batch_size=train_config['batch_size'], shuffle=True)
    if train_config['model_name']=="LSTM":
        model = LSTM().to(device)
        #数据集构建
        tokenizer=jieba.lcut_for_search
        
    

    elif train_config['model_name']=="Bert_biLSTM":
        model = Bert_biLSTM(pretrained_path=train_config['pretrained_path']).to(device)
        tokenizer = BertTokenizer.from_pretrained(train_config['pretrained_path'])

        train_len=len(train_dataloader.dataset)
    #模型训练
    for epoch in range(train_config['epochs']):
        model.train()
        pre=[]
        true=[]
        train_pre = 0
        for texts, labels in tqdm(train_dataloader):
            true += labels
            labels = labels.to(device)
            encoded_input = tokenizer(texts,  max_length=train_config['max_length'], padding='max_length', truncation=True, return_tensors="pt")
            input_ids, attention_masks, labels = encoded_input['input_ids'].to(device), encoded_input['attention_mask'].to(device), labels.to(device)
            logits = model(input_ids, attention_masks)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pre+=list(logits.argmax(1).to('cpu').numpy())
            train_pre += (logits.argmax(1) == labels).sum()
        print("train epoch:{}, accuracy: {}, f1: {}".format(epoch, train_pre/train_len, metrics.f1_score(true,pre)))
        bert_test(model, epoch, test_dataloader, tokenizer, device, train_config)
    bert_val(model, val_dataloader, tokenizer, device, train_config)

def bert_test(model, epoch, test_dataloader, tokenizer, device, train_config):
    test_length = len(test_dataloader.dataset)
    model.eval()
    pre=[]
    true=[]
    test_pre = 0
    for texts, labels in tqdm(test_dataloader,leave=False):
        true+=labels
        labels = labels.to(device)
        encoded_input = tokenizer(texts,  max_length=train_config['max_length'], padding='max_length', truncation=True, return_tensors="pt")
        input_ids, attention_masks, labels = encoded_input['input_ids'].to(device), encoded_input['attention_mask'].to(device), labels.to(device)
        logits = model(input_ids, attention_masks)
        test_pre += (logits.argmax(1) == labels).sum()
        pre += list(logits.argmax(1).to('cpu').numpy())
    print("test epoch:{}, accuracy: {}, f1: {}".format(epoch, test_pre/test_length,metrics(true, pre)))

def bert_val(model, val_dataloader, tokenizer, device, train_config):
    val_length = len(val_dataloader.dataset)
    model.eval()
    val_pre = 0
    pre=[]
    true=[]
    for texts, labels in tqdm(val_dataloader,leave=False):
        labels = labels.to(device)
        encoded_input = tokenizer(texts,  max_length=train_config['max_length'], padding='max_length', truncation=True, return_tensors="pt")
        input_ids, attention_masks, labels = encoded_input['input_ids'].to(device), encoded_input['attention_mask'].to(device), labels.to(device)
        logits = model(input_ids, attention_masks)
        val_pre += (logits.argmax(1) == labels).sum()
        pre += list(logits.argmax(1).to('cpu').numpy())
    print("val  accuracy: {}, f1: {}".format(val_pre/val_length), metrics.f1_score(true,pre))


if __name__ == "__main__":
    random_seed=100
    torch.manual_seed(random_seed)
    #train model
    train(train_config)
    