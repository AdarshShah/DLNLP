import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torchtext.vocab import GloVe
from torchtext.data.utils import get_tokenizer
from torch.utils.data import Dataset
import math
import numpy as np
import random
import os

def set_seed(seed = 42):
    '''
        For Reproducibility: Sets the seed of the entire notebook.
    '''

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    # Sets a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(1)

device = 'cuda:0'
glove = GloVe("6B")
myVocab = [f'number{i}' for i in range(6)] + ['+','-','*','/'] + ['sos','eos']
dim=64
nheads=8

tokenizer = get_tokenizer("basic_english")

def collatefun(batch):
    X = []
    Y = []
    Z = []
    for x,y,z in batch:
        for xx in x:
            xx = str.lower(xx)
            if xx in myVocab:
                X.append([1, myVocab.index(xx)])
            else: 
                X.append([0, glove.stoi[xx]] if xx in glove.stoi.keys() else [0, glove.stoi['unk']])
        for yy in y:
            yy = str.lower(yy)
            if yy in myVocab:
                Y.append([1, myVocab.index(yy)])
            else: 
                Y.append([0, glove.stoi[yy]] if yy in glove.stoi.keys() else [0, glove.stoi['unk']])
        Z = [ [1, myVocab.index(str.lower(zz))] for zz in z ]
    return X, Y, Z

def testcollatefun(batch):
    X = []
    Y = []
    Z = []
    A = []
    for x,y,z, a in batch:
        for xx in x:
            xx = str.lower(xx)
            if xx in myVocab:
                X.append([1, myVocab.index(xx)])
            else: 
                X.append([0, glove.stoi[xx]] if xx in glove.stoi.keys() else [0, glove.stoi['unk']])
        for yy in y:
            yy = str.lower(yy)
            if yy in myVocab:
                Y.append([1, myVocab.index(yy)])
            else: 
                Y.append([0, glove.stoi[yy]] if yy in glove.stoi.keys() else [0, glove.stoi['unk']])
        Z = z
        A = a
    return X, Y, Z, A

def em2sen(seq):
    res = ""
    for s in seq:
        if s[0]==0:
            res = res +' '+ glove.itos[s[1]]
        else:
            res = res+' '+myVocab[s[1]]
    return res

def jumble_numbers(sen1, sen2):
    '''
    suppose sen : 'w1 number0 w2 w3 number1'
    '''
    tokens = tokenizer(sen1)
    tokens2 = tokenizer(sen2)
    replace_with = np.random.permutation(np.arange(len(tokens2)//2 + 1))

    indices = { f'number{i}' : [j for j, token in enumerate(tokens) if token == f'number{i}'] for i in range(0,5) }
    for i in range(len(replace_with)):
        for idx in indices[f'number{i}']:
            tokens[idx] = f'number{replace_with[i]}'
    sen1 = ' '.join(tokens)

    indices = { f'number{i}' : [j for j, token in enumerate(tokens2) if token == f'number{i}'] for i in range(0,5) }
    for i in range(len(replace_with)):
        for idx in indices[f'number{i}']:
            tokens2[idx] = f'number{replace_with[i]}'

    sen2 = ' '.join(tokens2)
    return sen1, sen2

class A5aDataset(Dataset):

    def __init__(self, datapath) -> None:
        super(A5aDataset, self).__init__()
        dataframe = pd.read_excel(datapath)
        self.P = dataframe['Description']
        self.Q = dataframe['Question']
        self.E = dataframe['Equation']

    def __len__(self):
        return len(self.E)
    
    def __getitem__(self, index):
        sen1, sen2 = jumble_numbers(self.P[index], self.E[index])
        return tokenizer(sen1), tokenizer(self.Q[index]), tokenizer('sos '+sen2+' eos')

class A5aTestDataset(Dataset):

    def __init__(self, datapath) -> None:
        super(A5aTestDataset, self).__init__()
        dataframe = pd.read_excel(datapath)
        self.P = dataframe['Description']
        self.Q = dataframe['Question']
        self.E = dataframe['Input Numbers']
        self.A = dataframe['Output']

    def __len__(self):
        return len(self.E)
    
    def __getitem__(self, index):
        return tokenizer(self.P[index]), tokenizer(self.Q[index]), self.E[index], self.A[index]

class MyEmbedding(nn.Module):

    def __init__(self) -> None:
        super(MyEmbedding, self).__init__()
        glove.vectors[glove.stoi['unk']] = torch.zeros(300)
        self.gloveEmbedding = nn.Embedding.from_pretrained(glove.vectors, freeze=True)
        self.emlin = nn.Linear(300,dim, bias=False)
        # self.gloveEmbedding = nn.Embedding(len(glove), dim)
        self.myVocabEmbedding = nn.Embedding(len(myVocab), dim)
        self.myVocabEmbedding.weight.data.uniform_(-1,1)
        # self.gloveEmbedding.weight.data.uniform_(-1,1)

    def forward(self, sequence):
        embeddings = []
        for seq in sequence:
            if seq[0] == 0:
                embeddings.append(self.emlin(self.gloveEmbedding(torch.LongTensor([seq[1]]).to(device))))
            else:
                embeddings.append(self.myVocabEmbedding(torch.LongTensor([seq[1]]).to(device)))
        return torch.stack(embeddings).permute(1,0,2)

class PositionalEncoder(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = self.dropout(x + self.pe[:x.size(0)])
        return x

def op(op1, op2, operand):
    if operand=='+':
        return op1+op2
    if operand=='-':
        return abs(op1-op2)
    if operand=='/' and op2!=0:
        return op1/op2
    if operand=='*':
        return op1*op2
    return op1

def compute(sen, fields):
    try:
        sen = tokenizer(sen)
        sen = sen[1:-1]
        stack = []
        for word in sen:
            if word in ['+','-','*','/']:
                stack.append(word)
            elif stack[-1] in ['+','-','*','/']:
                stack.append(fields[word])
            else:
                op1 = stack.pop()
                operand = stack.pop()
                op2 = fields[word]
                stack.append(op(op1, op2, operand))

        return stack[-1]
    except:
        return 0

def gen_fields(sen):
    words = tokenizer(sen)
    fields = { f'number{i}':0 for i in range(6) }
    for i,word in enumerate(words):
        try:
            fields[f'number{i}']=int(word)
        except:
            pass
    return fields