# GRANLP

## Overview
Master MASH 2023 
NLP project : Toxic comment classification

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
the files used are all ```data/```, the names of the files used are in : 




## Models available

Have a look at the file `src/tests/test_run.py` for instructions on how to write your tests. You can run your tests as follows:

```
kedro test
```

To configure the coverage threshold, go to the `.coveragerc` file.



