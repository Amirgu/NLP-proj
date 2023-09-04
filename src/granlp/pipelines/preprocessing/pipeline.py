"""
This is a boilerplate pipeline 'preprocessing'
generated using Kedro 0.18.12
"""

from kedro.pipeline import Pipeline, pipeline,node
from .nodes import preprocess_train, preprocess_test

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline( [ node(
                func=preprocess_train,
                inputs=["train_comments"],
                outputs= "train_comments_processed",
                name="Processing_train" )


                ,node(
                func=preprocess_test,
                inputs=["test_comments"],
                outputs= "test_comments_processed",
                name="Processing_test" ) ],

                
                
                )
