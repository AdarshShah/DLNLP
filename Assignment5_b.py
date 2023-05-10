from symbol import except_clause
from turtle import forward
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from Assignment5_b_utils import A5aDataset, A5aTestDataset, collatefun, MyEmbedding, compute, em2sen, gen_fields, myVocab, PositionalEncoder, dim, testcollatefun, nheads
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
from tqdm import tqdm

trainpath = '/home/adarsh/DLNLP/datasets/Assignment5/ArithOpsTrain.xlsx'
testpath = '/home/adarsh/DLNLP/datasets/Assignment5/ArithOpsTestData1.xlsx'
device = 'cuda:0'
batch_size = 8
batch=0
mode='train'

dataset = A5aDataset(datapath=trainpath)
testdataset = A5aTestDataset(datapath=testpath)
freq = {}
try:
    for _,_, Z in dataset:
        for z in Z:
            if z not in freq.keys():
                freq[z] = 1
            else:
                freq[z] += 1
except:
    pass
weights = torch.Tensor([freq[z]+1 if z in freq.keys() else 1 for z in myVocab]).to(device)
weights = weights.sum()/weights
print(weights)
train, valid = random_split(dataset, [len(dataset)-100, 100])
train_dataloader = DataLoader(dataset, batch_size=1, collate_fn=collatefun, sampler=SubsetRandomSampler(train.indices))
valid_dataloader = DataLoader(valid, batch_size=1, collate_fn=collatefun)
test_dataloader = DataLoader(testdataset, batch_size=1, collate_fn=testcollatefun)

myEmbedding = MyEmbedding().to(device)

class Encoder(nn.Module):

    def __init__(self, myEmbedding) -> None:
        super(Encoder, self).__init__()
        self.embedding = myEmbedding
        self.pos = PositionalEncoder(dim, 200)
        self.gru = nn.GRU(dim, 256, bidirectional=False, batch_first=True)
        self.transformer = nn.Transformer(d_model=dim, nhead=nheads, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=dim*4, batch_first=True, norm_first=True)
        encoderLayer = nn.TransformerEncoderLayer(d_model=dim, nhead=8, dim_feedforward=dim*4, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoderLayer, num_layers=4)

    def forward(self, x, y):
        x = self.pos(self.embedding(x))
        y = self.pos(self.embedding(y))
        src_mask = torch.zeros((x.shape[1], x.shape[1])).bool().to(device)
        tgt_mask = torch.zeros((y.shape[1], y.shape[1])).bool().to(device)
        z = torch.concat((x,y),dim=1)
        # x,_ = self.gru(x)
        # y,_ = self.gru(y)
        return self.encoder(z), self.transformer(src=x, tgt=y, src_mask=src_mask, tgt_mask=tgt_mask)

class Decoder(nn.Module):

    def __init__(self, myEmbedding) -> None:
        super(Decoder, self).__init__()
        self.embedding = MyEmbedding()
        self.pos = PositionalEncoder(dim, 20)
        # self.gru = nn.GRU(300, 256, bidirectional=False, batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model=dim, nhead=nheads, dim_feedforward=dim*4, batch_first=True, norm_first=True)
        self.transdecoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=6)
        self.lin = nn.Linear(dim, len(myVocab), bias=False)
    
    def forward(self, z, enc_out):
        z = self.pos(self.embedding(z))
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(z.shape[1]).to(device)
        h = self.transdecoder(tgt=z, memory=enc_out, tgt_mask=tgt_mask)
        return self.lin(h).squeeze(0), h

class MyTranslator(nn.Module):

    def __init__(self, myEmbedding) -> None:
        super(MyTranslator, self).__init__()
        self.embedding = MyEmbedding()
        self.pos = PositionalEncoder(dim, 500)
        self.gru = nn.GRU(dim, dim, bidirectional=False, batch_first=True)
        self.transformer = nn.Transformer(d_model=dim, nhead=8, num_encoder_layers=8, num_decoder_layers=8, dim_feedforward=dim*4, batch_first=True)
        self.lin = nn.Linear(dim, len(myVocab), bias=False)
    
    def forward(self, x, y, z):
        x = x+y
        tgt_mask = self.transformer.generate_square_subsequent_mask(len(z)).to(device)
        # x,_ = self.gru(self.embedding(x))
        x = self.pos(self.embedding(x))
        # z,_ = self.gru(self.embedding(z))
        z = self.pos(self.embedding(z))
        return self.lin(self.transformer(src=x, tgt=z, tgt_mask=tgt_mask)).squeeze(0)


encoder = Encoder(myEmbedding).to(device)
decoder = Decoder(myEmbedding).to(device)
myTranslator = MyTranslator(myEmbedding).to(device)
encoder_optim = torch.optim.Adam(encoder.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
decoder_optim = torch.optim.Adam(decoder.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
myTranslator_optim = torch.optim.Adam(myTranslator.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
file = open("/home/adarsh/DLNLP/5_b.log",'a')

if mode=='test':
    encoder.load_state_dict(torch.load('/home/adarsh/DLNLP/models/encoder_66.pt'))
    decoder.load_state_dict(torch.load('/home/adarsh/DLNLP/models/decoder_66.pt'))

    testres = []
    with torch.no_grad():
        count = 0
        total = 0
        for x,y,z,a in tqdm(test_dataloader):
            _, memory = encoder(y,x)
            idx = [myVocab.index('sos')]
            loss = 0
            while myVocab[idx[-1]] != 'eos' and len(idx) < 11:
                z_  = [ (1, zz) for zz in idx]
                logits, dec_out = decoder(z_, memory)
                idx.append(torch.argmax(logits[-1]).item())
            z_  = [ (1, zz) for zz in idx]
            fields = gen_fields(z)
            result = compute(em2sen(z_), fields)
            if result==a:
                count += 1
            total += 1
            # print(result)
            testres.append(result)
    print(f'Test Accuracy : {count/total}')
    pd.DataFrame({'Adarsh':testres}).to_excel('/home/adarsh/DLNLP/datasets/Assignment5/adarsh_66.xlsx')

if mode=='train':
    loss=0
    for ep in range(100):
        with tqdm(train_dataloader) as tepoch:
            for x,y,z in tepoch:
                idx = [zz[1] for zz in z[1:]]        
                _, memory = encoder(y,x)                
                logits, dec_out = decoder(z[:-1], memory)
                loss += F.cross_entropy(input=logits, target=torch.LongTensor(idx).to(device), weight=weights)
                batch+=1
                if batch%batch_size==0:
                    loss = loss/batch_size
                    tepoch.set_postfix({'loss':loss.item()})
                    tepoch.refresh()
                    encoder_optim.zero_grad()
                    decoder_optim.zero_grad()
                    loss.backward()
                    encoder_optim.step()
                    decoder_optim.step()
                    loss=0
        with torch.no_grad():
            total = 0
            count = 0
            for x,y,z in valid_dataloader:
                _, memory = encoder(y,x)
                idx = [myVocab.index('sos')]
                loss = 0
                while myVocab[idx[-1]] != 'eos' and len(idx) < 11:
                    z_  = [ (1, zz) for zz in idx]
                    logits, dec_out = decoder(z_, memory)
                    idx.append(torch.argmax(logits[-1]).item())
                z_  = [ (1, zz) for zz in idx]
                file.write(em2sen(z)+ '||' + em2sen(z_) + '\n')
                if(em2sen(z) == em2sen(z_)):
                    count += 1
                total += 1
                file.flush()
            valid_accuracy = count/total
            file.write(f'Batch : {batch//batch_size} || Valid Accuracy : {valid_accuracy}\n')
            file.flush()

            torch.save(encoder.state_dict(), f'/home/adarsh/DLNLP/models/encoder_{int(100*valid_accuracy)}.pt')
            torch.save(decoder.state_dict(), f'/home/adarsh/DLNLP/models/decoder_{int(100*valid_accuracy)}.pt')