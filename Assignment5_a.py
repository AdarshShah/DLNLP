import math
import os
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler, random_split, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchtext.data.utils import get_tokenizer
from torchtext.transforms import BERTTokenizer
from torchtext.vocab import GloVe, build_vocab_from_iterator
from tqdm import tqdm

writer = SummaryWriter('/home/adarsh/DLNLP/logs/assgn5')

datapath = '/home/adarsh/DLNLP/datasets/Assignment2/dataset.csv'
testpath = '/home/adarsh/DLNLP/datasets/Assignment2/test.csv'
word_toeknizer = get_tokenizer('basic_english')

vocabpath = 'https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt'
device = 'cuda:0'
log = open('/home/adarsh/DLNLP/assgn5_log.txt', 'w')
glove = GloVe('6B')

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
testdataset = ReviewDataset(testpath)

def generate_vocab(dataset):
    print('> generating vocab file for word piece tokenizer')
    try:
        with open(vocabpath, 'a') as file:
            for X, Y in tqdm(dataset):
                words = word_toeknizer(X)
                [file.write(word+'\n') for word in words]
    except:
        pass


def set_seed(seed=42):
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


def max_len(dataset, tokenizer):
    mx = 0
    try:
        for X, Y in dataset:
            words = word_toeknizer(X)
            mx = max(len(words), mx)
    except:
        pass
    print('> max len of review : ', mx)
    return mx


def yield_tokens(dataset, tokenizer):
    try:
        for X, Y in tqdm(dataset):
            yield tokenizer(X)
    except:
        pass



# tokenizer = None
# try:
#     tokenizer = BERTTokenizer(vocab_path=vocabpath, return_tokens=True)
# except:
#     generate_vocab(dataset)
#     tokenizer = BERTTokenizer(vocab_path=vocabpath, return_tokens=True)
tokenizer = get_tokenizer('basic_english')
# print('> building vocab')
# vocab = build_vocab_from_iterator(yield_tokens(dataset, tokenizer), min_freq=2, specials=['<unk>'])
# vocab.set_default_index(vocab['<unk>'])
# print('> vocab length : ', len(vocab))
# maxlen = max_len(dataset, tokenizer)

def trunc(seq):
    if len(seq)>512:
        return seq[:256] + seq[-256:]
    return seq

def collate_fun(batch):
    X = torch.nn.utils.rnn.pad_sequence([torch.tensor(trunc([glove.stoi[word] if word in glove.stoi.keys(
    ) else glove.stoi['unk'] for word in tokenizer(b[0])])) for b in batch], True, glove.stoi['unk']).long().to(device)
    Y = torch.stack([torch.ones((1,)) if b[1] == 'positive' else torch.zeros(
        (1,)) for b in batch]).to(device)
    return X, Y


train_dataset, valid_dataset = random_split(
    dataset, [9*len(dataset)//10, len(dataset)//10])

train_sampler = SubsetRandomSampler(train_dataset.indices)
valid_sampler = SubsetRandomSampler(valid_dataset.indices)

train_dataloader = DataLoader(
    dataset, batch_size=8, sampler=train_sampler, collate_fn=collate_fun)
valid_dataloader = DataLoader(
    dataset, batch_size=8, sampler=valid_sampler, collate_fn=collate_fun)
test_dataloader = DataLoader(
    testdataset, batch_size=8, collate_fn=collate_fun)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class Classifier(nn.Module):

    def __init__(self, ntokens, embed_dim, maxlen, nhead, nlayers) -> None:
        super(Classifier, self).__init__()
        glove.vectors[glove.stoi['unk']] = torch.zeros(300)
        print(
            f'embed_dim : {embed_dim}\t nhead : {nhead}\t nlayers : {nlayers}')
        self.embedding = nn.Embedding.from_pretrained(
            embeddings=glove.vectors, freeze=False)
        self.embed_lin = nn.Linear(300, embed_dim, bias=False)
        self.pos = PositionalEncoding(embed_dim, max_len=maxlen, dropout=0)
        encoder = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=4*embed_dim, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder, nlayers)
        self.gru = nn.GRU(embed_dim, embed_dim//2, bidirectional=True, batch_first=True)
        self.lin2 = nn.Linear(embed_dim, 1)

    def forward(self, seq):
        hid = self.embed_lin(self.embedding(seq))
        hid, _ = self.gru(hid)
        hid = self.encoder(hid)
        return self.lin2(hid.mean(dim=1)), torch.sigmoid(self.lin2(hid.mean(dim=1)).detach())


model = Classifier(ntokens=len(glove), embed_dim=256,
                   maxlen=512, nhead=16, nlayers=6).to(device)

print('> training')
g = 0
optim = torch.optim.Adam(model.parameters())
for ep in range(5):
    count = 0
    with tqdm(train_dataloader) as tepoch:
        for X, Y in tepoch:
            tepoch.set_description(f'Epoch {ep}')
            logits, pred = model(X)
            optim.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, Y)
            loss.backward()
            optim.step()
            # scheduler.step()
            tepoch.set_postfix({'loss': loss.item()})
            writer.add_scalar(f'Train_Loss_{ep}', loss.item(), g)
            g += 1
            tepoch.refresh()
            torch.cuda.empty_cache()
            if g % 250 == 0:
                sum = 0
                with torch.no_grad():
                    for X, Y in valid_dataloader:
                        _, pred = model(X)
                        pred = torch.round(pred)
                        sum += (pred == Y).float().sum().item()
                        torch.cuda.empty_cache()
                    writer.add_scalar(f'Valid_Accuracy',
                                      sum/len(valid_dataset), g)
                    print('> valid accuracy : ', sum/len(valid_dataset))

                sum = 0
                with torch.no_grad():
                    for X, Y in test_dataloader:
                        _, pred = model(X)
                        pred = torch.round(pred)
                        sum += (pred == Y).float().sum().item()
                        torch.cuda.empty_cache()
                    writer.add_scalar(f'Valid_Accuracy',
                                      sum/len(testdataset), g)
                    print('> test accuracy : ', sum/len(testdataset))
