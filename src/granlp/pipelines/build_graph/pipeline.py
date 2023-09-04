"""
This is a boilerplate pipeline 'build_graph'
generated using Kedro 0.18.12
"""

from kedro.pipeline import Pipeline, pipeline,node
from .nodes import build_graph

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([node(
                func=build_graph,
                inputs=["train_comments_processed","test_comments_processed","params:parameters"],
                outputs= ["train_dt","val_dt","test_dt"],
                name="zeb")])
