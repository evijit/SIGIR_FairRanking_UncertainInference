from __future__ import unicode_literals, print_function, division
from collections import defaultdict
import time
import random
import torch
from io import open
import glob
import os
import unicodedata
import string
import csv
import numpy as np
import zipfile
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import time 
from torch.utils.data import DataLoader
import sklearn.metrics
import numpy as np


def seed_everything(seed=100):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    

class KIM_CNN(torch.nn.Module):
    def __init__(self, nwords, emb_size, num_filters, window_sizes, ntags, emb_type=None, dropout=0.5):
        super(KIM_CNN, self).__init__()
        
        if emb_type=='glove':
            self.embedding= nn.Embedding.from_pretrained(torch.FloatTensor(id_to_glove_vec))
        elif emb_type=='fasttext':
            self.embedding= nn.Embedding.from_pretrained(torch.FloatTensor(id_to_fasttext_vec))
        elif emb_type=='word2vec':
        	self.embedding= nn.Embedding.from_pretrained(torch.FloatTensor(id_to_word2vec_vec))            
        else:
            self.embedding= nn.Embedding(nwords, emb_size, padding_idx=w2i['<PAD>'])
            nn.init.uniform_(self.embedding.weight, -0.25,0.25)

        self.embedding.weight.requires_grad= True
        
        # applying 1 layer convolution 
        
        self.convs= nn.ModuleList([nn.Conv1d(in_channels=emb_size,out_channels=num_filters,kernel_size=window_size) 
                    for window_size in window_sizes])
        
        # self.conv2d= nn.Conv2d(in_channels=emb_size,out_channels=num_filters, kernel_size=window_size)
    
        self.relu= nn.ReLU()
        self.dropout= nn.Dropout(dropout)
        self.projection_layer= nn.Linear(in_features=num_filters*len(window_sizes), out_features=ntags, bias=True)
        nn.init.xavier_uniform_(self.projection_layer.weight)
    
    def forward(self, words, return_activations=False):
        embeds=self.embedding(words) # BATCH_SIZE * n*_words * dim
        embeds=embeds.permute(0,2,1) # BATCH_SIZE * dim *n_words
        h=[conv(embeds) for conv in self.convs] # [BATCH_SIZE * n_filters*(n_words- window_size +1)] * len(WINDOW_SIZE)
        h=[self.relu(h1) for h1 in h]  
        h=[h1.max(dim=2)[0] for h1 in h]
        h= torch.cat(h,1)
        h= self.dropout(h)
        out= self.projection_layer(h)
        return out

# edic = {}
# gdic = {}

def load(infile):
    edic = {}
    gdic = {}

    races = set([])
    eths = set([])
    #data = []
    with open(infile, encoding='utf-8', errors='ignore') as csvfile:
        reader = csv.reader(csvfile, delimiter = '\t', quotechar='"')
        first = True
        for row in tqdm(reader):
            if first:
                first = False
                continue

            last = row[9]
            first = row[10]
            middle = row[11]
            city = row[14]
            state = 'NC'
            zipcode = row[16]
            race = row[25]
            races.add(race)
            ethnic = row[26]
            eths.add(ethnic)
            gender = row[28]
            status = row[4]
            party = row[27]
            age = row[29]
            
            if ethnic == 'HL':
                rcat = 'Latino'
            if ethnic == 'NL':
                rcat = race
            if rcat not in ['U', 'O','M', ' '] and len(rcat)>0:
                if rcat == 'B':
                    rcat = 'Black'
                if rcat == 'A':
                    rcat = 'Asian'
                if rcat == 'I':
                    rcat = 'NativeAm'
                if rcat == 'W':
                    rcat = 'White'
                if rcat not in edic:
                    edic[rcat]=[]
                edic[rcat].append(first+" "+last)
                
            if gender not in gdic:
                gdic[gender]=[]
            gdic[gender].append(first+" "+last)
    
    return gdic, edic
    

def get_data():
    w2i = {}
    t2i = {}
    w2i["<unk>"] = 1
    w2i['<PAD>'] = 0

    data = []
    for gender in ['M', 'F']:
        label = gender
        if label not in t2i:
            t2i[label] = len(t2i)
        #with open(file, 'r') as f:
        for line in gdic[label]:
            chars= list(line.lower().strip())
            for char in chars:
                if char not in w2i:
                    w2i[char]= len(w2i)
            data.append((chars, label))
            
    random.shuffle(data)
   
    return data, w2i, t2i


def get_batches(batch):
    label = torch.tensor([t2i[entry[1]] for entry in batch])
    max_len= max([len(entry[0]) for entry in batch])
    text =[[w2i[i] for i in entry[0]]+ [w2i['<PAD>'] for i in range(max_len-len(entry[0]))] for entry in batch]
    return text, label


def run_model(model, steps):
    criterion = torch.nn.CrossEntropyLoss()
    weights = [len(data)/len(gdic[eth]) for eth in t2i] 
    class_weights = torch.FloatTensor(weights).cuda()
    criterion_weights = torch.nn.CrossEntropyLoss(weight=class_weights)
    #criterion_weights = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_model=None
    best_epoch=0
    best_f1 =0

    longtype = torch.LongTensor
    floattype= torch.FloatTensor
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        model.cuda()
        longtype = torch.cuda.LongTensor
        floattype= torch.cuda.FloatTensor


    for ITER in range(ITERS):
        
        train_data = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True,
                      collate_fn=get_batches)
        dev_data = DataLoader(dev, batch_size= BATCH_SIZE, shuffle=False,
                      collate_fn=get_batches)

        train_loss = 0.0
        train_pred = []
        train_true = []
        train_accs = []
        train_f1s  = []
        
        start = time.time()
        dev_pred = []
        dev_true = []
        
        dev_loss=0.0

        for index, (text,labels) in enumerate(train_data):
            text= torch.tensor(text).type(longtype)
            labels= labels.type(longtype)
            scores=model(text)
            predict=[score.argmax().item() for score in scores]
            train_true.extend(labels.cpu().numpy())
            train_pred.extend(predict)
            train_accs.append(sklearn.metrics.accuracy_score(labels.cpu().numpy(),predict))
            train_f1s.append(sklearn.metrics.f1_score(labels.cpu().numpy(), predict, average='macro'))
            
            if index%steps==0:
                print("Train acc {}  Train F1 {} for {}/{}".format(round(np.mean(train_accs),3), round(np.mean(train_f1s),3),index, len(train_data)), end='\r')
                train_accs = []
                train_f1s  = []
            
            my_loss = criterion(scores, labels)
            train_loss += my_loss.item()
            optimizer.zero_grad()
            my_loss.backward()
            optimizer.step()

        print("Train iter {} loss/sent {} acc {} f1 {}".format(ITER, train_loss / len(train),np.round(sklearn.metrics.accuracy_score(train_true,train_pred),4),np.round(sklearn.metrics.f1_score(train_true, train_pred, average='macro'),4)))
        
        for index, (text,labels) in enumerate(dev_data):
            text= torch.tensor(text).type(longtype)
            labels= labels.type(longtype)
            scores=model(text)
            predict=[score.argmax().item() for score in scores]
            dev_true.extend(labels.cpu().numpy())
            dev_pred.extend(predict)
            
        print("Valid iter {} acc {} f1 {}".format(ITER, np.round(sklearn.metrics.accuracy_score(dev_true, dev_pred),4),np.round(sklearn.metrics.f1_score(dev_true, dev_pred, average='macro'),4)))
        f1 = sklearn.metrics.f1_score(dev_true, dev_pred, average='macro')
        
        if f1 > best_f1:
            best_f1 = f1
            best_model=model
            best_epoch= ITER

        # if np.mean(val_f1s)> best_dev_f1:
        #   best_dev_f1= np.mean(val_f1s)

    print("Best f1 score is ", best_f1)
        
    return model, best_f1, best_epoch

nc_dir = '/home/piotr/voter-records-nc/'
gdic, edic =  load(nc_dir+'ncvoter_Statewide.txt')

import sys

seed = 100
try:
    seed = int(sys.argv[1])
except Exception as e:
    seed = 100

hyperparams_dict = {}
hyperparams_dict['seed']       = seed
hyperparams_dict['classifier'] = 'gender'
hyperparams_dict['middle_name']= 'no'

seed_everything(hyperparams_dict['seed'])
data, w2i, t2i = get_data()

print(t2i)

train = data[0:int(0.7*len(data))]
dev   = data[int(0.7*len(data)):int(0.8*len(data))]
test  = data[int(0.8*len(data)):int(len(data))]

DROPOUT=0.3
EMB_SIZE   = 64
WIN_SIZE   = 3
FILTER_MAP = 100
BATCH_SIZE = 3200
WIN_SIZES=[3,4,5]
LEARNING_RATE= 0.0005
VOCAB_SIZE = len(w2i)
LABEL_SIZE = len(t2i)
ITERS = 10
STEPS = 5


def call_gender_inference(name, params_dict):
    model_path = '-'.join([elem+'_'+str(params_dict[elem]) for elem in sorted(params_dict)])+'.pt'
    best_model = torch.load(model_path, map_location=torch.device('cpu'))
    check = [(list(name),'M')]
    check_data = DataLoader(check, batch_size= 1, shuffle=False, collate_fn=get_batches)
    
    for text, label in check_data:
        text= torch.tensor(text).type(longtype)
        scores=best_model(text)
        predict=[score.argmax().item() for score in scores]
        return inv_tag2i[predict[0]]


def call_gender_inference_batch(names, params_dict):
    model_path = '-'.join([elem+'_'+str(params_dict[elem]) for elem in sorted(params_dict)])+'.pt'
    best_model = torch.load(model_path, map_location=torch.device('cpu'))
    check = [(list(name),'M') for name in names]
    check_data = DataLoader(check, batch_size= min(len(names),1000), shuffle=False, collate_fn=get_batches)
    
    labels_arr = []
    
    for text, label in check_data:
        text= torch.tensor(text).type(longtype)
        scores=best_model(text)
        predict_arr=[score.argmax().item() for score in scores]
        labels_arr.extend([inv_tag2i[predict] for predict in predict_arr])
   
    return labels_arr
    
    


train_flag = False
test_flag  = False

longtype = torch.LongTensor
floattype= torch.FloatTensor


# longtype = torch.cuda.LongTensor
# floattype= torch.cuda.FloatTensor

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

model = KIM_CNN(VOCAB_SIZE, EMB_SIZE, FILTER_MAP, WIN_SIZES, LABEL_SIZE, 'random',DROPOUT)

model_path = '-'.join([elem+'_'+str(hyperparams_dict[elem]) for elem in sorted(hyperparams_dict)])+'.pt'

inv_tag2i={}
for tag in t2i:
    inv_tag2i[t2i[tag]]=tag

model_dir = '/home/avijit/indeedproject/FairRanking/'

if train_flag:
    best_model, acc, f1= run_model(model,5)
    torch.save(best_model,model_dir+model_path)


if test_flag:
    print(model_dir)
    best_model = torch.load(model_dir+model_path)

    inv_tag2i={}
    for tag in t2i:
        inv_tag2i[t2i[tag]]=tag

    test_data  = DataLoader(test, batch_size= BATCH_SIZE, shuffle=False,
                          collate_fn=get_batches)
    test_true=[]
    test_pred=[]

    use_cuda = True

    for index, (text,labels) in enumerate(test_data):
#         if use_cuda:
#             longtype = torch.cuda.LongTensor
#             floattype= torch.cuda.FloatTensor

        text= torch.tensor(text).type(longtype)
        scores=best_model(text)
        predict=[score.argmax().item() for score in scores]
        test_true.extend([inv_tag2i[label] for label in labels.numpy()])
        test_pred.extend([inv_tag2i[label] for label in predict])
        

    print("Test acc {} f1 {}".format(np.round(sklearn.metrics.accuracy_score(test_true, test_pred),4),np.round(sklearn.metrics.f1_score(test_true, test_pred, average='macro'),4)))
    print(sklearn.metrics.classification_report(test_true, test_pred))
