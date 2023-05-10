from Assignment4_b import datapath, ReviewDataset, testpath
import torch
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import random
import os

writer = SummaryWriter('/home/adarsh/DLNLP/logs/assgn5')
device = 'cuda:1'
dataset = ReviewDataset(datapath)
testdataset = ReviewDataset(testpath)

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


from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

bert = AutoModel.from_pretrained("bert-base-uncased")


def collate_fun(batch):
    X = tokenizer([ str.lower(b[0]) for b in batch ], padding=True, truncation=True, return_tensors="pt").to(device)
    Y = torch.stack([ torch.ones((1,)) if b[1]=='positive' else torch.zeros((1,)) for b in batch ]).to(device)
    return X,Y

class Classifier(nn.Module):

    def __init__(self, model) -> None:
        super(Classifier, self).__init__()
        self.bert = model
        self.lin = nn.Linear(768, 1)
    
    def forward(self, seq):
        em = self.bert(**seq)
        return self.lin(em.last_hidden_state.detach().mean(dim=1)), torch.sigmoid(self.lin(em.last_hidden_state.detach().mean(dim=1)).detach())

train_dataset, valid_dataset = random_split(dataset, [ 9*len(dataset)//10, len(dataset)//10 ])

train_sampler = SubsetRandomSampler(train_dataset.indices)
valid_sampler = SubsetRandomSampler(valid_dataset.indices)

train_dataloader = DataLoader(dataset, batch_size=8, sampler=train_sampler, collate_fn=collate_fun)
valid_dataloader = DataLoader(dataset, batch_size=8, sampler=valid_sampler, collate_fn=collate_fun)
test_dataloader = DataLoader(
    testdataset, batch_size=8, collate_fn=collate_fun)

model = Classifier(bert).to(device)

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
