from torch.utils.data import Dataset
import torch



class Data_corpus(Dataset):
  def __init__(self, dataframe, vocab, tokenizer, max_len,lemma):
    super().__init__()
    assert 'cleaned_text' in list(dataframe.columns), "no column named cleaned_data"
    self.dataframe = dataframe
    self.vocab = vocab
    self.tokenizer =  tokenizer
    self.max_len = max_len
    self.lemmatizer = lemma
    self.sos = [vocab['<sos>']]
    self.eos = [vocab['<eos>']]
    self.pad = [vocab['<pad>']]
    self.cls = [vocab['<CLS>']]

  def __getitem__(self, index):

    text, label = self.dataframe.iloc[index][['cleaned_text', 'label_num']]
    text = [self.lemmatizer.lemmatize(word) for word in self.tokenizer(text)]
    #print(text)
    #text = [word for word in self.tokenizer(text)]
    #text = self.tokenizer(text)

    tokens = torch.LongTensor(self.cls + self.sos + self.vocab( text[:self.max_len-3] )+ self.eos + self.pad*(self.max_len - len( text[:self.max_len])-3))
    mask = tokens == self.vocab['<pad>']
    return tokens , mask, torch.tensor(label).long()


  def __len__(self):
    return len(self.dataframe)