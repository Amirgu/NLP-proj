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
The files used are all ```data/```, the names of the files used are in : `conf/local/`
Upload the datesets as follows :

 * Set up the training set in '''data/02_intermediate/train.csv'''
 
 * Set up the test set in '''data/02_intermediate/test.csv'''
 
 * Set up the test labels in '''data/02_intermediate/test_labels.csv'''


## Models available

Have a look at the file `src/tests/test_run.py` for instructions on how to write your tests. You can run your tests as follows:

```
kedro test
```

To configure the coverage threshold, go to the `.coveragerc` file.



