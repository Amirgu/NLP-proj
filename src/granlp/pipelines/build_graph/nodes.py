import os
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

import ast 
from sklearn.model_selection import train_test_split
from .utils.Bert_processing import MultiLabelDataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


log = logging.getLogger(__name__)

def build_graph(train_data,test_data,parameters):
    if parameters['model'] == 'DistilBert':
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
        testing_params = {'batch_size': parameters['TRAIN_BATCH_SIZE'],
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
        print(X.shape, test_X.shape)

        vect = TfidfVectorizer(max_features=5000,stop_words='english')
        X_dtm = vect.fit_transform(X)
        text_X_dtm = vect.transform(test_X)
        
        return X_dtm,0,text_X_dtm