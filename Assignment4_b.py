import os
import random
import re

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import (DataLoader, Dataset, SubsetRandomSampler,
                              random_split)
from torch.utils.tensorboard import SummaryWriter
from torchtext.data import get_tokenizer
from torchtext.vocab import GloVe, FastText
from tqdm import tqdm

#Global Variables
datapath = '/home/adarsh/DLNLP/datasets/Assignment2/dataset.csv'
testpath = '/home/adarsh/DLNLP/datasets/Assignment2/test.csv'
device = 'cuda:0'
writer = SummaryWriter('/home/adarsh/DLNLP/logs/assgn4/')
global_step = 0

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

vocab = GloVe('6B')
word_toeknizer = get_tokenizer('basic_english')

def sen_tokenizer(review:str):
    pattern = "(\.|\?|\!|<br />)+"
    seq =  re.split(pattern, review)
    ret = []
    for sen in seq:
        if not re.match(pattern, sen) and len(sen)!=0:
            sen = str.strip(sen)
            words = word_toeknizer(sen)
            while len(words) > 70:
                ret.append(' '.join(words[:70]))
                words = words[70:]
            if len(words) > 0:
                ret.append(' '.join(words))
    return ret

class Word2Vec(nn.Module):

    def __init__(self, vocab:GloVe, tokenizer, freeze=False):
        super(Word2Vec, self).__init__()
        self.vocab = vocab
        self.words = set(self.vocab.itos)
        self.tokenizer = tokenizer
        self.vectors = nn.Embedding.from_pretrained(self.vocab.vectors, freeze=freeze)
  
    def forward(self, seq:torch.Tensor):
        return self.vectors(seq)
    
    def sen2vec(self, sentence:str):
        tokens = self.tokenizer(sentence)
        return torch.LongTensor([self.vocab.stoi[str.lower(word)] if str.lower(word) in self.words else 201534 for word in tokens]).to(device)


class HANblock(nn.Module):

    def __init__(self, embed_dim, hidden_dim) -> None:
        super(HANblock, self).__init__()
        self.rnn1 = nn.GRU(embed_dim, hidden_dim//2, bidirectional=False, batch_first=True)
        self.rnn2 = nn.GRU(embed_dim, hidden_dim//2, bidirectional=False, batch_first=True)
        self.uw = nn.Linear(hidden_dim, 1)
    
    def forward(self, w):
        # w : B x L x embed_dim
        fwd_hidden, _ = self.rnn1(w)
        # hidden : B x L x hidden_dim
        bkd_hidden,_ = self.rnn2(torch.flip(w, dims=(1,)))
        # hidden : B x L x hidden_dim
        hidden = torch.cat((fwd_hidden, torch.flip(bkd_hidden, dims=(1,))), dim=2)
        # hidden : B x L x 2*hidden_dim
        alpha = F.softmax(self.uw(hidden).squeeze(2), dim=1).unsqueeze(1)
        # alpha : B x 1 x L
        sen = torch.bmm(alpha, hidden).squeeze(1)
        # sen : B x 2*hidden_dim
        return sen

class HAN(nn.Module):

    def __init__(self) -> None:
        super(HAN, self).__init__()
        self.word2Vec = Word2Vec(vocab, word_toeknizer, freeze=False)
        self.word_attn = HANblock(300, 300)
        self.sen_attn = HANblock(300, 300)
        self.lin1 = nn.Linear(300, 600)
        self.lin2 = nn.Linear(600, 1)
    
    def forward(self, seq):
        # seq : str | review
        seq = sen_tokenizer(seq)
        # seq : S | sentence
        seq = [ self.word2Vec.sen2vec(sen) for sen in seq ]
        # seq : S x * | Batched unpadded 
        seq = pad_sequence(seq, batch_first=True, padding_value=201534)
        # seq : S x L
        embed = self.word2Vec(seq)
        # embed : S x L x embed_dim
        sen = self.word_attn(embed)
        # sen : S x embed_dim
        sen = sen.unsqueeze(0)
        # sen : 1 x S x embed_dim
        doc = self.sen_attn(sen)
        # doc : 1 x embed_dim
        return torch.sigmoid((self.lin2(torch.tanh(self.lin1(doc.squeeze())))))

class ReviewDataset(Dataset):

    def __init__(self, datapath) -> None:
        super(ReviewDataset, self).__init__()
        df = pd.read_csv(datapath)
        self.X = df['review']
        self.Y = df['sentiment']
    
    def __getitem__(self, index):
        return self.X[index], self.Y[index]
    
    def __len__(self):
        return len(self.X)

dataset = ReviewDataset(datapath)
test = ReviewDataset(testpath)
train, valid = random_split(dataset, [9*len(dataset)//10, len(dataset)//10])

if __name__=='__main__':
    model = HAN().to(device)
    optim = torch.optim.Adam(model.parameters())

    #model('A rating of "1" does not begin to express how dull, depressing and relentlessly bad this movie is.')

    for ep in range(2):
        try:
            with tqdm(train) as tepoch:
                for review, sentiment in tepoch:
                    optim.zero_grad()
                    pred = model(review)
                    y = torch.ones((1,)).to(device) if sentiment == 'positive' else torch.zeros((1,)).to(device)
                    loss = F.binary_cross_entropy(pred, y)
                    loss.backward()
                    optim.step()
                    tepoch.set_postfix({'loss':loss.item()})
                    writer.add_scalar('Train Loss', loss.item(), global_step)
                    global_step+=1
                    tepoch.refresh()
                
                    if global_step in [10000, 15000, 20000, 25000, 30000]:
                        with torch.no_grad():
                            acc = 0
                            with tqdm(valid) as tepoch:
                                for review, sentiment in tepoch:
                                    try:
                                        pred = model(review).round()
                                        y = 1 if sentiment == 'positive' else 0
                                        acc += 1 if pred.item() == y else 0
                                        tepoch.set_postfix({'Valid Accuracy': acc/len(valid)})
                                        tepoch.refresh()
                                    except:
                                        import pdb; pdb.set_trace()
                            writer.add_scalar('Valid Accuracy', acc/len(valid), global_step)
            
                        with torch.no_grad():
                            acc = 0
                            count = 0
                            try:
                                with tqdm(test) as tepoch:
                                    for review, sentiment in tepoch:
                                        try:
                                            pred = model(review).round()
                                            y = 1 if sentiment == 'positive' else 0
                                            acc += 1 if pred.item() == y else 0
                                            count += 1
                                            tepoch.set_postfix({'Test Accuracy': acc/count})
                                            tepoch.refresh()
                                        except:
                                            import pdb; pdb.set_trace()
                            except:
                                pass
                            writer.add_scalar('Test Accuracy', acc/count, global_step)
        except:
            pass

        with torch.no_grad():
            acc = 0
            with tqdm(valid) as tepoch:
                for review, sentiment in tepoch:
                    try:
                        pred = model(review).round()
                        y = 1 if sentiment == 'positive' else 0
                        acc += 1 if pred.item() == y else 0
                        tepoch.set_postfix({'Valid Accuracy': acc/len(valid)})
                        tepoch.refresh()
                    except:
                        import pdb; pdb.set_trace()
            writer.add_scalar('Valid Accuracy', acc/len(valid), global_step)
            acc = 0
            count = 0
            try:
                with tqdm(test) as tepoch:
                    for review, sentiment in tepoch:
                        try:
                            pred = model(review).round()
                            y = 1 if sentiment == 'positive' else 0
                            acc += 1 if pred.item() == y else 0
                            count += 1
                            tepoch.set_postfix({'Test Accuracy': acc/count})
                            tepoch.refresh()
                        except:
                            import pdb; pdb.set_trace()
            except:
                pass
            writer.add_scalar('Test Accuracy', acc/count, global_step)
