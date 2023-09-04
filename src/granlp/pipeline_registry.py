"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from granlp.pipelines import preprocessing, loading, models, eval


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines()
    pipelines["__default__"] = sum(pipelines.values())

    preprocessing_pipe = preprocessing.create_pipeline()
    loading_pipe = loading.create_pipeline()
    models_pipe = models.create_pipeline()
    eval_pipe = eval.create_pipeline()
    return{"__default__": preprocessing_pipe, "preprocessing": preprocessing_pipe, "loading": loading_pipe,"models":models_pipe,"eval":eval_pipe}
