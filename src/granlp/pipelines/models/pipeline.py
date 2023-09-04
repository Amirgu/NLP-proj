"""
This is a boilerplate pipeline 'models'
generated using Kedro 0.18.12
"""

from kedro.pipeline import Pipeline, pipeline,node
from .nodes import train

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([node(
                func=train,
                inputs=["train_dt","val_dt","test_dt","train_comments_processed","test_comments_processed","params:parameters"] ,
                outputs = "submission_tfidf_logreg",
                name="Training")])
