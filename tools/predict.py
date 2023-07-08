import pathlib, sys
__dir__ = pathlib.Path(__file__).parent
sys.path.append(str(__dir__.parent))

from typing import List, Any, Tuple
from numpy.typing import NDArray

import pickle, ast
import pandas as pd
from data import load_tranformer, StatTest

class Predictor:
    def __init__(self, phase: str, problem: str,  model_name: str="CatBoostClassifier", postfix="_2", fold="4") -> None:
        self.work_dir = __dir__.parent/f"samples/{phase}/{problem}"

        # drift detector
        self.stattest = StatTest()

        # model
        if fold:
            f = (self.work_dir/"model"/f"{model_name}{postfix}"/fold/"model.pickle").open(mode="rb")
        else:
            f = (self.work_dir/"model"/f"{model_name}{postfix}"/"model.pickle").open(mode="rb")
        self.model = pickle.load(f)

        # get transformer
        self.category_transformer = load_tranformer(self.work_dir/"transformer", f"category_transformer{postfix}")
        self.numeric_transformer = load_tranformer(self.work_dir/"transformer", f"numeric_transformer{postfix}")

        # get importance feature
        if fold:
            self.important_features = (self.work_dir/"model"/f"{model_name}{postfix}"/fold/"feature_columns.txt").read_text().split("\n")
        else:
            self.important_features = (self.work_dir/"model"/f"{model_name}{postfix}"/"feature_columns.txt").read_text().split("\n")

        # get feature config
        self.feature_config = {
            "numeric_columns": [col for col in self.important_features if col in self.numeric_transformer.columns],
            "category_columns": [col for col in self.important_features if col in self.category_transformer.columns]
        }

        # refenence data
        self.reference_data = pd.read_parquet(str(self.work_dir/"train"/"raw_train.parquet"), engine="pyarrow")[self.important_features]

    def __call__(self, df: pd.DataFrame) -> Tuple[bool, NDArray]:
        df = df[self.important_features]
        # is_drift, drift_score = self.detect_drift(df)
        is_drift = True
        X = self.transform(df)
        return is_drift, self.predict(X)

    def predict(self, X: NDArray):
        return self.model.predict(
            X,
            thread_count=3,
            verbose=False,
            task_type='CPU'
        )
    
    def detect_drift(self, df: pd.DataFrame) -> Tuple[bool, float]:
        is_drift, drift_score = self.stattest.detect_drift_data(
            reference_df=self.reference_data,
            current_df=df,
            feature_config=self.feature_config

        )
        return is_drift, drift_score
    
    def transform(self, df: pd.DataFrame):
        for column in df.columns:
            if column in self.category_transformer.columns:
                df = self.category_transformer.transform(df, column=column)
            elif column in self.numeric_transformer.columns:
                df = self.numeric_transformer.transform(df, column=column)
            else:
                df = df.drop(columns=[column])
        
        return df.values
    
