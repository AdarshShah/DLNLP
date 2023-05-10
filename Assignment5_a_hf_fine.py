from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import random

os.environ["WANDB_DISABLED"] = "true"
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", use_fast=True)
device = 'cuda:0'

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

bert = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2).to(device)
print(bert.eval())

def prep_dataset(path="/home/adarsh/DLNLP/datasets/Assignment2/dataset.csv", split=0.1):
    dataset = load_dataset("csv",data_files=path,split="train")
    if split > 0 :
        dataset = dataset.train_test_split(test_size=split, shuffle=False)
    dataset = dataset.map(lambda x : { 'text': x['review'], 'label' : 0 if x['sentiment']=='negative' else 1 })
    dataset = dataset.map(lambda x : tokenizer(x['text'], padding="max_length", truncation=True))
    return dataset

metric = evaluate.load("accuracy")

def compute_metrics(logits, labels):
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

traindatapath = "/home/adarsh/DLNLP/datasets/Assignment2/dataset.csv"
testpath = "/home/adarsh/DLNLP/datasets/Assignment2/test.csv"
train_dataset = prep_dataset(traindatapath, 0.1)
test_dataset = prep_dataset(testpath, 0)


training_args = TrainingArguments(
    output_dir="/home/adarsh/DLNLP/5_a_checkpoint",
    evaluation_strategy="epoch",
    num_train_epochs=5.0,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=8,
    save_steps=100,
    logging_steps=250)

trainer = Trainer(
    model=bert,
    args=training_args,
    train_dataset=train_dataset['train'],
    eval_dataset=train_dataset['test'],
    compute_metrics=compute_metrics
)

# trainer.train()

def acc(model, dataloader):
    accuracies = []
    for batch in tqdm(dataloader):
        input_ids = torch.stack(batch['input_ids']).to(device).permute(1,0)
        attention_mask = torch.stack(batch['attention_mask']).to(device).permute(1,0)
        label = batch['label'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=label)
        logits = outputs['logits'].detach().cpu().numpy()
        labels = label.cpu().numpy()
        accuracies.append(compute_metrics(logits, labels)['accuracy'])
    print(np.mean(accuracies))


valid = train_dataset['test']
valid_dataloader = DataLoader(valid, batch_size = 32)
test_dataloader = DataLoader(test_dataset, batch_size = 32)

for m in range(100, 1501, 100):
    model = AutoModelForSequenceClassification.from_pretrained(f'/home/adarsh/DLNLP/5_a_checkpoint/checkpoint-{m}').to(device)
    print(f'valid accuracy > {acc(model, valid_dataloader)}')
    print(f'test accuracy > {acc(model, test_dataloader)}')


