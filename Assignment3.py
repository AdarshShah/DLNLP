import numpy as np
import pandas as pd
import spacy
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from torch.nn import GRU, ModuleList, Linear
from torch.nn.functional import relu
from torch.optim import Adagrad
import gc
from torch.nn.utils.rnn import pad_sequence

nlp = spacy.load('/home/adarsh/DLNLP/spacy/glove_840b_300d')
filepath = '/home/adarsh/DLNLP/datasets/Assignment2/dataset.csv'
testpath = '/home/adarsh/DLNLP/datasets/Assignment2/test.csv'
device = 'cuda:0'

class ReviewDataset(Dataset):
    def __init__(self, filepath, nlp) -> None:
        super(ReviewDataset, self).__init__()
        self.nlp = nlp
        dataset = pd.read_csv(filepath)
        print("Dataset Preparation")
        docs = [ nlp(review) for review in tqdm(dataset['review'])]
        self.X = [ torch.stack([ torch.tensor(token.vector) for token in doc ]).to(device) for doc in tqdm(docs) ]
        self.Y = torch.tensor(np.where(dataset['sentiment']=='negative', 0, 1)).to(device)
    
    def __getitem__(self, index):
        return self.X[index], self.Y[index].float()
    
    def __len__(self):
        return len(self.X)

dataset = ReviewDataset(filepath, nlp)
test_dataset = ReviewDataset(testpath, nlp)

def collate_fn(batch):
    Y = torch.stack([ b[1] for b in batch])
    X = pad_sequence([ b[0] for b in batch ], padding_value=0)
    return (X.to(device),Y.to(device))


class RecurrentClassifier(torch.nn.Module):

    def __init__(self) -> None:
        super(RecurrentClassifier, self).__init__()
        self.grus = ModuleList([
            GRU(300, 300, bidirectional=True),
            GRU(300, 300, bidirectional=True)
        ])
        self.alpha = torch.nn.parameter.Parameter(torch.randn((1,)))
        self.beta = torch.nn.parameter.Parameter(torch.randn((1,)))
        self.feed_forward= ModuleList([
            Linear(300, 900),
            Linear(900, 1),
        ])
    
    def forward(self, input):
        _, h1 = self.grus[0](input)
        _, h2 = self.grus[1](input)
        v = torch.exp(self.alpha) + torch.exp(self.beta)
        h = (torch.exp(self.alpha)*h1[0] + torch.exp(self.beta)*h2[-1])/v
        return torch.sigmoid(self.feed_forward[2](torch.relu(self.feed_forward[1](relu(self.feed_forward[0](h))))))
    
    def sen2vec(self, input):
        _, h1 = self.grus[0](input)
        _, h2 = self.grus[1](input)
        return self.alpha*h1[0] + self.beta*h2[-1]

train, valid = random_split(dataset, [len(dataset)-2*len(dataset)//10 , 2*len(dataset)//10])
train_dataloader = DataLoader(train, batch_size=32, collate_fn=collate_fn)
valid_dataloader = DataLoader(valid, batch_size=128, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=128, collate_fn=collate_fn)
model = RecurrentClassifier()
model.to(device)

optim = Adagrad(model.parameters())
loss = 0
losses = []
accuracy = 0
loss_fn = torch.nn.MSELoss()
first = True
print("Training:")
for ep in range(50):
    with tqdm(train_dataloader) as tepoch:
        for i, (X, y) in enumerate(tepoch):
            tepoch.set_description(f'Epoch {ep}')
            pred = model(X).squeeze()
            loss = loss_fn(pred, y)
            if first:
                print(f'First Training Loss : {loss.item()}')
            optim.zero_grad()
            loss.backward()
            optim.step()
            tepoch.set_postfix({'loss':loss.item()})
            first=False
    gc.collect()
    torch.cuda.empty_cache()
    valid_losses = 0
    with tqdm(valid_dataloader) as tepoch:
        for i, (X, y) in enumerate(tepoch):
            pred = model(X).squeeze()
            pred = torch.round(pred)
            valid_losses += (pred==y).nonzero().sum().item()
    valid_losses /= len(valid)
    print(f'Valid Accuracy : {valid_losses}')

print('Testing:')
test_losses = 0
with tqdm(test_dataloader) as tepoch:
    for i, (X, y) in enumerate(tepoch):
        pred = model(X).squeeze()
        pred = torch.round(pred)
        test_losses += (pred==y).nonzero().sum().item()
test_losses /= len(test_dataset)
print(f'Test Accuracy : {test_losses}')