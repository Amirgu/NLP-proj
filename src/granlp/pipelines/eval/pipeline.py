"""
This is a boilerplate pipeline 'eval'
generated using Kedro 0.18.12
"""

from kedro.pipeline import Pipeline, pipeline,node
from .nodes import eval

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([node(
                func=eval,
                inputs=["test_comments_processed","test_dt","params:parameters"] ,
                outputs = "submission",
                name="ze")])
