import pandas as pd
import nltk




def prep(file  ='spam_ham_dataset.csv'):
    
    nltk.download('punkt')
    nltk.download('wordnet')

    data = pd.read_csv(file)
    data['subject'] = data['text'].str.split('\r', n=1, expand=True)[0].str.split(':', n=1, expand=True)[1]
    data['text'] =  data['text'].str.split('\r', n=1, expand=True)[1]

    special_characters = r'[^a-zA-Z\d\s]+'
    single_characters = r'\b[a-zA-Z]\b'
    tabs_newline_pattern = r'[\t\n\r]'
    multiple_spaces = r'\s+'
    number = r'\d+'


    data['cleaned_text'] = data['text'].str.replace(special_characters, '', regex=True)
    data['cleaned_text'] = data['cleaned_text'].str.replace(single_characters, '', regex=True)
    data['cleaned_text'] = data['cleaned_text'].str.replace(tabs_newline_pattern, ' ', regex=True)
    data['cleaned_text'] = data['cleaned_text'].str.replace(multiple_spaces, ' ', regex=True)
    data['cleaned_text'] = data['cleaned_text'].str.replace(number, '[QUANTITY] ', regex=True)
