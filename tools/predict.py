import pathlib, sys
__dir__ = pathlib.Path(__file__).parent
sys.path.append(str(__dir__.parent))

from typing import List, Any
from numpy.typing import NDArray

import pickle
import pandas as pd
from data import load_tranformer

class Predictor:
    def __init__(self, phase: str, problem: str,  model_name: str="CatBoostClassifier") -> None:
        self.work_dir = __dir__.parent/f"samples/{phase}/{problem}"

        # dataset
        self.category_transformer = load_tranformer(self.work_dir/"transformer", "category_transformer")
        self.numeric_transformer = load_tranformer(self.work_dir/"transformer", "numeric_transformer")

        # model
        f = (self.work_dir/"model"/model_name/"model.pickle").open(mode="rb")
        self.model = pickle.load(f)

    def __call__(self, df: pd.DataFrame) -> Any:
        X = self.transform(df)
        return self.predict(X)

    def predict(self, X: NDArray):
        return self.model.predict(
            X,
            thread_count=5,
            verbose=True,
            task_type='CPU'
        )
    
    def transform(self, df: pd.DataFrame):
        for column in df.columns:
            if column in self.category_transformer.columns:
                df = self.category_transformer.transform(df, column=column)
            elif column in self.numeric_transformer.columns:
                df = self.numeric_transformer.transform(df, column=column)
            else:
                df = df.drop(columns=[column])
        
        return df.values
    
