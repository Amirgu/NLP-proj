"""
This is a boilerplate pipeline 'models'
generated using Kedro 0.18.12
"""
import torch 

from torch import cuda
from .models.distilbert import DistilBERTClass, SaveBestModel,loss_fn
import logging
log = logging.getLogger(__name__)
import tqdm
import torch.nn as nn
from sklearn.metrics import precision_score,roc_auc_score
import numpy as np 
from sklearn.linear_model import LogisticRegression
import ast 
from sklearn.metrics import accuracy_score
from .models.lstm import model_add
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
import logging 
log = logging.getLogger(__name__)
def train_bert(training_loader,val_loader,test_loader,params):
    """
    Training Distilbert 
    """
    device = torch.device('cuda' if cuda.is_available() else 'cpu')
    print(f"Current device: {device}")
    model = DistilBERTClass()
    model= nn.DataParallel(model)
    model.to(device)
    
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=params['LEARNING_RATE'])
    
    best_model = SaveBestModel()
    
    for epoch in range(params['EPOCHS']):
        model.train()
        for _,data in tqdm.tqdm(enumerate(training_loader, 0)):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
    
            outputs = model(ids, mask, token_type_ids)
    
            optimizer.zero_grad()
            loss = loss_fn(outputs, targets)
            if _%5000==0:
                log.info(f'Epoch: {epoch}, Loss:  {loss.item()}')
            
            loss.backward()
            optimizer.step()
            
        
        pred = []
        true = []
        with torch.no_grad():
            total_loss = []
            
            for i,data in tqdm.tqdm(enumerate(val_loader, 0)):
                model.eval()
                ids = data['ids'].to(device, dtype = torch.long)
                mask = data['mask'].to(device, dtype = torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
                targets = data['targets'].to(device, dtype = torch.float)
                outputs = model(ids, mask, token_type_ids)
                val_loss = loss_fn(outputs, targets)
               
                total_loss.append( val_loss.item())
                true += targets.cpu().numpy().tolist()
                pred += outputs.cpu().numpy().tolist()
            true = np.array(true)
            pred = np.array(pred)
            pred = pred >= 0.5
            best_model( np.mean(total_loss),
        epoch, model, optimizer 
    )
            for i, name in enumerate(['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']):
                log.info(f"{name} precision {precision_score(true[:, i], pred[:, i])}")
            log.info(f"Evaluate loss {np.mean(total_loss)}")
            
def add_feature(X, feature_to_add):
    '''
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    '''
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')        

def train(training_loader,val_loader,test_loader,train_data,test_data,params):
    """ 
    Training models 
    """
    
    if params['model'] == 'DistilBert':
        log.info('Started Training DistilBert')
        train_bert(training_loader,val_loader,test_loader,params)
        return test_data
    if params['model'] == 'Tfidf':
        cols_target = ['obscene','insult','toxic','severe_toxic','identity_hate','threat']
        
        X_dtm = training_loader
        test_X_dtm = test_loader
        train_data['labels'] = train_data['labels'].map(ast.literal_eval)
        
        Y = np.array(list(train_data['labels'].to_numpy()))
        submission = test_data.drop(columns=['comment_text'])
        logreg = LogisticRegression(C=12.0)
        
        for i,label in enumerate(cols_target):
            log.info('... Processing {}'.format(label))
            y= Y[:,i]
            # train the model using X_dtm & y
            logreg.fit(X_dtm, y)
            # compute the training accuracy
            y_pred_X = logreg.predict(X_dtm)
            
            log.info('Training accuracy is {}'.format(accuracy_score(y, y_pred_X)))
            test_y = logreg.predict(test_X_dtm)
            test_y_prob = logreg.predict_proba(test_X_dtm)[:,1]
            submission[label] = test_y_prob
            X_dtm = add_feature(X_dtm, y)
            test_X_dtm = add_feature(test_X_dtm, test_y)
            log.info('Shape of X_dtm is now {}'.format(X_dtm.shape))
            log.info('The submission are available at : data/08_reporting/submission__tfidf_logreg.csv')
        return submission
    if params['model'] == 'LSTM':
       
        
        model = model_add()
        log.info('model summary:')
        log.info(model.summary())
        train_data['labels'] = train_data['labels'].map(ast.literal_eval)
        y = np.array(list(train_data['labels'].to_numpy()))
        
        early = EarlyStopping(monitor="val_loss", mode="min", patience=20)
        log.info('Started Training LSTM')
        model.fit(training_loader, y, batch_size=params['TRAIN_BATCH_SIZE'], epochs=params['EPOCHS'], validation_split=0.1, callbacks=early)
        log.info('Saving the best model to :  data/01_raw/model.keras ')
        model.save('data/01_raw/model.keras')
        
        return test_data
