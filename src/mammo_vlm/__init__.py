"""Pure-Python pieces of the grounded mammography VLM pipeline.

Everything in here runs without a GPU or the dataset, which is the point:
it's the stuff that can silently produce bad training data or wrong metrics,
so it gets unit tests. Training/inference live in the notebooks.
"""
