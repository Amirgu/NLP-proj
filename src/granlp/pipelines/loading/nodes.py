"""
This is a boilerplate pipeline 'loading'
generated using Kedro 0.18.12
"""
import random
import numpy as np
import pickle as pkl

from math import log
from sklearn import svm
from nltk.corpus import wordnet as wn

import sys

from transformers import DistilBertTokenizer
import pandas as pd 
import ast
from torch.utils.data import Dataset, DataLoader 
import logging
import torch
import csv 
import pandas 
import keras
import ast 
from sklearn.model_selection import train_test_split
from .utils.Bert_processing import MultiLabelDataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
log = logging.getLogger(__name__)

def loading(train_data,test_data,parameters):
    """Loading datasets for the three models :

    Args: 
        train_data: training pandas dataframe (catalog:train_comments_processed)
        test_data: test pandas dataframe (catalog:test_comments_processed)
    Returns : 
        training_loader,val_loader,test_loader: a file to feed the model following the model chosen
    
    """
    if parameters['model'] == 'Distilbert':
        log.info("Model chosen : DistilBert")
        train_data,val_data= train_test_split(train_data,test_size = 0.1)
        tokenizer = DistilBertTokenizer.from_pretrained('data/02_intermediate/distilbert-base-uncased', truncation=True, do_lower_case=True)
        training_set = MultiLabelDataset(train_data, tokenizer,parameters['MAX_LEN'])
        val_set = MultiLabelDataset(val_data, tokenizer,parameters['MAX_LEN'], eval_mode = False)
        test_set = MultiLabelDataset(test_data, tokenizer, parameters['MAX_LEN'], eval_mode = True)
        
        train_params = {'batch_size': parameters['TRAIN_BATCH_SIZE'],
                    'shuffle': True,
                    'num_workers': parameters['NUM_WORKERS']
                    }
        testing_params = {'batch_size': parameters['TEST_BATCH_SIZE'],
                       'shuffle': False,
                       'num_workers': parameters['NUM_WORKERS']
                        }
        
        log.info('LOADING TRAINING SET')
        training_loader = DataLoader(training_set, **train_params)
        log.info('LOADING VALIDATION SET')
        val_loader = DataLoader(val_set, **testing_params)
        log.info('LOADING TEST SET')
        test_loader = DataLoader(test_set, **testing_params)
        return training_loader,val_loader,test_loader
        
    elif parameters['model'] == 'Tfidf':
        
        log.info("Model chosen : Tfid+Logreg")
        X = train_data.comment_text
        test_X = test_data.comment_text
        
        

        vect = TfidfVectorizer(max_features=5000,stop_words='english')
        log.info('Vectorizing Training set')
        X_dtm = vect.fit_transform(X)
        log.info('Vectorizing Test set')
        text_X_dtm = vect.transform(test_X)
        
        return X_dtm,0,text_X_dtm
    elif parameters['model'] == 'LSTM':

        log.info("Model chosen : LSTM")
        sentiment = train_data['comment_text'].values
        
        tokenizer = Tokenizer(num_words=20000)
        tokenizer.fit_on_texts(list(sentiment))
        
        log.info('LOADING TRAINING SET')
        seq = tokenizer.texts_to_sequences(sentiment)
        pad = sequence.pad_sequences(seq, maxlen=100)
        log.info('LOADING TEST SET')
        test = test_data['comment_text'].values
        test_seq = tokenizer.texts_to_sequences(test)
        test_pad = sequence.pad_sequences(test_seq, maxlen=100)

        return pad ,0,test_pad