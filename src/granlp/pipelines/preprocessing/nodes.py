"""
This is a boilerplate pipeline 'preprocessing'
generated using Kedro 0.18.12
"""
from kedro.pipeline import node
import re

import logging
log = logging.getLogger(__name__)




def text_preprocessing(text):
    """
    Text preprocessing removing slangs, and shortcuts and various other things 
    """
    
    text = text.lower()
    text = re.sub(r'(@.*?)[\s]', ' ', text) #remove mentions or usernames in the text.
    text = re.sub(r'[0-9]+' , '' ,text) # removes any sequence of digits

    text = re.sub(r'\s([@][\w_-]+)', '', text).strip() 
    text = re.sub(r'&amp;', '&', text) #replaces the HTML entity '&' with an ampersand '&'
    text = re.sub(r'\s+', ' ', text).strip() #extra whitespace
    text = text.replace("#" , " ")
    # replace 
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    encoded_string = text.encode("ascii", "ignore") #encodes the text into ASCII format
    decode_string = encoded_string.decode() #decodes the ASCII-encoded text back into a string.
    return decode_string

def preprocess_train(train_data):
    """ 
    Loading and Applying Preprocessing training data 
    """
    log.info('Preprocessing Training Data')
    train_data.drop(['id'], inplace=True, axis=1)
    train_data['labels'] = train_data.iloc[:, 1:].values.tolist()
    train_data.drop(train_data.columns.values[1:-1].tolist(), inplace=True, axis=1)
    train_data["comment_text"].map(text_preprocessing)
    train_data["comment_text"] = train_data["comment_text"].str.lower()
    train_data["comment_text"] = train_data["comment_text"].str.replace("\xa0", " ", regex=False).str.split().str.join(" ")
    log.info("HEAD OF Training data:")
    log.info(train_data.head(10))

    #print(f"The shape of our dataset {len(train_data["comment_text"])}")
    
    log.info(f"The shape of our labels: {train_data['labels'].shape}")

    
    return train_data

## remove brackets from labels 
def preprocess_test(test_data):
    """ 
    Loading and Applying Preprocessing training data 
    """
    log.info('Preprocessing test Data')
    test_data["comment_text"].map(text_preprocessing)
    log.info("HEAD OF TESTING DATA:")
    log.info(test_data.head(10))
    log.info(f"The shape of our labels: {test_data['comment_text'].shape}")
    return test_data