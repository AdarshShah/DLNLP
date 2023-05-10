import numpy as np
from nltk.corpus import wordnet as wn, words
from torchtext.data import get_tokenizer
from random import sample
from torch.utils.data import Dataset, DataLoader
import pandas as pd

def synonymns(word: str):
    # synonymns = [word]
    # try:
    #     [synonymns := synonymns +
    #         synset.lemma_names('eng') for synset in wn.synsets(word)]
    # except:
    #     pass
    # synonymns = list(set(synonymns))
    # np.random.shuffle(synonymns)
    # try:
    #     return ' '.join(synonymns)
    # except:
    return str(word)


def pertube(sen: str):
    basic_english = get_tokenizer('basic_english')
    tokens = []
    for token in basic_english(sen):
        if np.random.rand() < 0.9:
            tokens.append(token)
        if np.random.rand() < 0.1:
            tokens.append(sample(words.words(), 1)[0])
    sen = ' '.join(tokens)
    return sen

class EngDataset(Dataset):

    def __init__(self, path='/home/adarsh/DLNLP/datasets/Assignment6/EnglishDictionary.csv') -> None:
        super(EngDataset, self).__init__()
        self.dictionary = pd.read_csv(path)
    
    def __getitem__(self, index):
        return pertube(self.dictionary['Definition'][index]), synonymns(self.dictionary['Word'][index])
    
    def __len__(self):
        return len(self.dictionary['Word'])
