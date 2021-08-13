import torch
from torchtext.legacy.data import Field
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from torchtext.vocab import Vectors, GloVe
import numpy as np 
import pandas as pd 
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Pair:
    def __init__(self, sr, trg):
        self.sent = sr
        self.label = trg


def remove_stopword(w):
    stop_word = '@#$%^&**()[]/<->\;:{}"'
    for i in stop_word:
        w = w.replace(i, '')
    return w


def get_data():
    dataset = pd.read_excel('dataset/data_text_sentiment_vad_2.xlsx', sheet_name='TRAIN')
    sentence = dataset['SENTENCE']
    label = dataset['LABEL']
    return sentence, label


def get_datasets(lenght_sen,batch_size):
    train = []
    sent, label = get_data()
    df = pd.DataFrame({'sentence': sent, 'sentiment': label})
    source = Field(tokenize='basic_english', lower=True)
    sentence = df['sentence'].apply(lambda x: source.preprocess(x))
    source.build_vocab(sentence, vectors=GloVe(name='6B', dim=300))
    source_vocab = source.vocab
    word_embeddings = source.vocab.vectors
    vocab_size = len(source.vocab)
    for i in range(len(sentence)):
        e = list(map(lambda x: source_vocab.stoi[x], sentence[i]))
        if len(e) <= lenght_sen:
            e = np.array(e + (lenght_sen - len(e)) * [1]).reshape((lenght_sen))
        else:
            e = np.array(e[:lenght_sen - 1] + [1]).reshape((lenght_sen))
        train.append(Pair(e, label[i]))
    sent = [feature.sent for feature in train]
    label = [feature.label for feature in train]
    data_train = TensorDataset(torch.LongTensor(sent), (torch.LongTensor(label)))
    data_train, data_val = train_test_split(data_train,test_size=0.3,shuffle=True)
    data_train = DataLoader(data_train, batch_size=batch_size, drop_last=True,shuffle=True)
    data_val = DataLoader(data_val, batch_size=batch_size, drop_last=True,shuffle=True)
    return source, vocab_size, word_embeddings, data_train, data_val


if __name__ == '__main__':
    TEXT, vocab_size, word_embeddings, train_iter, val_iter = get_datasets(32,256)

