# GRANLP

## Overview
Master MASH 2023 
NLP Course project : Jigsaw Toxic Comment classification https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

Models available :
  * DistilBERT
  * TF-IDF Vectorization + Logistic Regression Classification
  * Bidirectional LSTM


## Rules and guidelines

In order to get the best out of the template:

* Don't remove any lines from the `.gitignore` file we provide
* Make sure your results can be reproduced by following a data engineering convention
* Don't commit data to your repository
* Don't commit any credentials or your local configuration to your repository. Keep all your credentials and local configuration in `conf/local/`

## How to install dependencies

Declare any dependencies in `src/requirements.txt` for `pip` installation and `src/environment.yml` for `conda` installation.

To install them, run:

```
pip install -r src/requirements.txt
```
# Set up the project
## How to run the project

You can run your Kedro project with:

```
git clone 
cd ./NLP-proj
```
## Set up the datasets
The files used are all in ```data/```, the names of the files used and their locations are in : `conf/base/catalog.yml`
Upload the datesets as follows :

 * Set up the training set in `data/02_intermediate/train.csv`
 * Set up the test set in `data/02_intermediate/test.csv`
 * Set up the test labels in `data/02_intermediate/test_labels.csv`

Choose the model and configure the parameters in : `conf/base/parameters.yml`
## Set up the DistilBert model : 
Install the model DistilBERT (distilbert-base-uncase) from : (https://huggingface.co/docs/transformers/model_doc/distilbert)
Place it in  `data/02_intermediate/train.csv`

## To run the models:
After configuring and choosing the model you want you run the following: 
* For Preprocessing the dataset :
 ```
 kedro run -p preprocessing 
 ```
* For Loading the dataset :
 ```
 kedro run -p loading 
 ```
* For training the model :
 ```
 kedro run -p models 
 ```

* For getting the evaluation on the test set(only for LSTM and DistilBert as the submission will already be available after latter command for the Tfidf model) :
  
 ```
 kedro run -p eval 
 ```
The submissions will be available depending on the model as stated in the file `conf/base/catalog.yml`

## Directory : 

Our code will all be available in `src/granlp/pipelines/`

For example : 

For the preprocessing code : `src/granlp/pipelines/preprocessing/nodes.py`



