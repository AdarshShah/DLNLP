'''
Task 1 : Create English Reverse Dictionary
'''

from transformers import T5Tokenizer, T5ForConditionalGeneration
from Assignment6_utils import EngDataset, DataLoader
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('/home/adarsh/DLNLP/logs/assgn6')
hf_model = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(hf_model, model_max_length=512)
model = T5ForConditionalGeneration.from_pretrained(hf_model).cuda(0)
optim = torch.optim.Adam(model.parameters(), lr=1e-4)

dataset = EngDataset()
train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

global_step = 0
with tqdm(train_dataloader) as tepoch:
    for x,y in tepoch:
        try:
            # print(f'inputs {x}')
            # print(f'labels {y}')
            x = tokenizer(x, return_tensors='pt', padding=True, truncation=True).input_ids
            y = tokenizer(y, return_tensors='pt', padding=True, truncation=True).input_ids
            loss = model(input_ids=x.cuda(0), labels=y.cuda(0)).loss
            optim.zero_grad()
            loss.backward()
            writer.add_scalar('train loss', loss.item(), global_step)
            global_step += 1
            optim.step()
            outputs = model.generate(x.cuda(0))
            # print(f'outputs {tokenizer.batch_decode(outputs, skip_special_tokens=True)}')
        except:
            pass

# model.save_pretrained('/home/adarsh/DLNLP/models/assignment_6')
model = T5ForConditionalGeneration.from_pretrained('/home/adarsh/DLNLP/models/assignment_6').cuda(0)
outputs = model.generate(tokenizer("The place where doctor works and patients visit", return_tensors='pt', padding=True, truncation=True).input_ids.cuda(0))


