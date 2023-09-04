"""
This is a boilerplate pipeline 'eval'
generated using Kedro 0.18.12
"""

import torch
from torch import cuda
import torch.nn as nn
from .models.distilbert import DistilBERTClass
import tqdm
import keras 


def test(model,test_loader):
    all_test_pred = []
    model.eval()
    
    device = torch.device('cuda')
    
    with torch.inference_mode():
    
        for _, data in tqdm.tqdm(enumerate(test_loader, 0)):


            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            outputs = model(ids, mask, token_type_ids)
            probas = torch.sigmoid(outputs)

            all_test_pred.append(probas)
    return all_test_pred 




def eval(test_data,test_loader,params):
    """
    Evaluating Models and returning Submissons
    """
    if params['model'] == 'DistilBert':
        device = torch.device('cuda')
        model = DistilBERTClass()
        model= nn.DataParallel(model)
        checkpoint = torch.load('data/01_raw/best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
    
        all_test_pred = test(model,test_loader)
        all_test_pred = torch.cat(all_test_pred)
        submit_df = test_data.copy()
        submit_df.drop("comment_text", inplace=True, axis=1)
        label_columns = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        for i,name in enumerate(label_columns):
    
            submit_df[name] = all_test_pred[:, i].cpu()
            print(submit_df.head())
        submit_df['id'] = test_data['id']
    if params['model'] == 'LSTM':
        model = keras.models.load_model("data/01_raw/model.keras")
        y_test = model.predict([test_loader], batch_size=1024, verbose=1)
        label_columns = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        submit_df = test_data.copy()
        submit_df.drop("comment_text", inplace=True, axis=1)
        submit_df[label_columns] = y_test
    return submit_df