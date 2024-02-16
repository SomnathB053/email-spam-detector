from preproc import prep
from dataset import Data_corpus
from model import SpamClassifier


import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from collections import Counter
import torchtext
from torchtext.data import get_tokenizer
from nltk.stem import WordNetLemmatizer
from torch.utils.data import DataLoader


data = prep()
data_80 = data.iloc[:int(0.8*len(data))][['label_num', 'cleaned_text']]
data_20 = data.iloc[int(0.8*len(data)):][['label_num', 'cleaned_text']]
corpus = []
wnl = WordNetLemmatizer()
count = 0
for i in data_80['cleaned_text']:
  count+=1
  try:
    corpus = corpus  + [''.join([wnl.lemmatize(word) for word in i]).strip()]
  except:
    print(count)
counter = Counter()
tokenizer = get_tokenizer('basic_english')
for sent in corpus:
  counter.update(tokenizer(sent))
vocab = torchtext.vocab.vocab(counter)
for sp_token, index in zip(['<unk>', '<pad>', '<sos>', '<eos>', '<CLS>'], [0,1,2,3,4]):
        vocab.insert_token(sp_token, index)
        vocab.set_default_index(0)


model_config = {
    'seq_len': 60,
    'd_model': 512,
    'batch_size': 128,
    'n_encode': 2,
    'n_head': 4,
    'ffn_dim': 1024,
    'dropout': 0.1
}



def train(EPOCHS, seq_len, batch_size, d_model, n_encode, n_head, ffn_dim, dropout):
    dataset_train = Data_corpus(data_80, vocab, tokenizer, seq_len, WordNetLemmatizer())
    loader_train = DataLoader(dataset_train, batch_size, shuffle=True)

    dataset_val = Data_corpus(data_20, vocab, tokenizer, seq_len, WordNetLemmatizer())
    loader_val = DataLoader(dataset_val, batch_size, shuffle=True)

    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Compute device is {device}')
    #MODEL SPECS
    model = SpamClassifier(len(vocab), seq_len, d_model, n_encode,n_head,ffn_dim,dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.000002)
    #criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion2 = nn.CrossEntropyLoss()

    len_loader_train = len(loader_train)
    len_loader_val = len(loader_val)

    for epoch in range(EPOCHS):
        print(f'Starting epoch -----[{epoch}]-----')
        avg_loss_train = 0
        avg_loss_val = 0
        model.train()
        with tqdm(loader_train) as tq:
            for _, e in enumerate(tq):
                tq.set_description(f'Batch number: {_}: ')
                tokens, masks, labels = e
                tokens = tokens.to(device)
                masks = masks.to(device)
                labels = labels.to(device)
                output = model(tokens, masks)
                loss_TRAIN = criterion2(output, labels)
                tq.set_postfix(current_loss= loss_TRAIN.item())
                avg_loss_train += loss_TRAIN.item()
            loss_TRAIN.backward()
            optimizer.step()
        model.eval()
        for _, e in enumerate(loader_val):
            with torch.no_grad():
                tokens, masks, labels = e
                tokens = tokens.to(device)
                masks = masks.to(device)
                labels = labels.to(device)
                output = model(tokens, masks)
                loss_VAL = criterion2(output, labels)
                avg_loss_val += loss_VAL.item()
    print(f'[Average TRAINING LOSS: {avg_loss_train/len_loader_train}]-----[Average VALIDATION LOSS: {avg_loss_val/len_loader_val}]')
    torch.save(model.state_dict(), 'weights.pt')


train(EPOCHS=75, **model_config)