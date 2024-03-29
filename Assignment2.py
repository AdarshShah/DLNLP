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

nlp = spacy.load('/home/adarsh/DLNLP/spacy/glove_840b_300d')
filepath = '/home/adarsh/DLNLP/datasets/Assignment2/dataset.csv'
testpath = '/home/adarsh/DLNLP/datasets/Assignment2/test.csv'
device = 'cuda:1'

class ReviewDataset(Dataset):
    def __init__(self, filepath, nlp) -> None:
        super(ReviewDataset, self).__init__()
        self.nlp = nlp
        dataset = pd.read_csv(filepath)
        print("Dataset Preparation")
        docs = [ nlp(review) for review in tqdm(dataset['review'])]
        self.X = [ torch.stack([ torch.tensor(token.vector) for token in doc ]) for doc in tqdm(docs) ]
        self.Y = torch.tensor(np.where(dataset['sentiment']=='negative', 0, 1))
    
    def __getitem__(self, index):
        return self.X[index].to(device), self.Y[index].float().to(device)
    
    def __len__(self):
        return len(self.X)

dataset = ReviewDataset(filepath, nlp)
test_dataset = ReviewDataset(testpath, nlp)

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
            Linear(300, 100),
            Linear(100, 1),
        ])
    
    def forward(self, input):
        _, h1 = self.grus[0](input)
        _, h2 = self.grus[1](input)
        h = self.alpha*h1[0] + self.beta*h2[-1]
        return torch.sigmoid(self.feed_forward[1](relu(self.feed_forward[0](h))))
    
    def sen2vec(self, input):
        _, h1 = self.grus[0](input)
        _, h2 = self.grus[1](input)
        return self.alpha*h1[0] + self.beta*h2[-1]

train, valid = random_split(dataset, [len(dataset)-len(dataset)//10 , len(dataset)//10])
dataloader = DataLoader(train, batch_size=1)
test_dataloader = DataLoader(test_dataset, batch_size=1)
model = RecurrentClassifier()
model.to(device)

optim = Adagrad(model.parameters())
loss = 0
losses = []
valid_x = [ dataset.X[i] for i in valid.indices]
valid_y = dataset.Y[valid.indices]
accuracy = 0
loss_fn = torch.nn.MSELoss()
first = True
print("Training:")
for ep in range(2):
    with tqdm(dataloader) as tepoch:
        for i, (X, y) in enumerate(tepoch):
            tepoch.set_description(f'Epoch {ep}')
            pred = model(X[0])
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
    pred = torch.stack([ model(x.to(device)).cpu() for x in valid_x])
    pred = torch.round(pred)
    pred = torch.reshape(pred, (pred.shape[0],))
    accuracy = (pred == valid_y).sum().item()/pred.shape[0]
    print(f'First Validation Accuracy : {accuracy}')

print('Testing:')
test_losses = []
with tqdm(test_dataloader) as tepoch:
    for i, (X, y) in enumerate(tepoch):
        pred = model(X[0])
        pred = torch.round(pred)
        test_losses.append(1 if pred==y else 0)
test_losses = np.array(test_losses).mean().item()
print(f'Test Accuracy : {test_losses}')