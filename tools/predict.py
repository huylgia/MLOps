import pathlib, sys
__dir__ = pathlib.Path(__file__).parent
sys.path.append(str(__dir__.parent))

from typing import List, Any, Tuple
from numpy.typing import NDArray

import pickle, ast
import pandas as pd
import numpy as np
from data import StatTest
from catboost import CatBoostClassifier, Pool

class Predictor:
    def __init__(self, phase: str, problem: str,  model_name: str="CatBoostClassifier") -> None:
        self.work_dir = __dir__.parent/f"samples/{phase}/{problem}"

        # drift detector
        self.stattest = StatTest()

        # model
        f = (self.work_dir/"model"/model_name/"model.pickle").open(mode="rb")
        self.model: CatBoostClassifier = pickle.load(f)

        # feature config
        feature_config = (self.work_dir/"train"/"features_config.json").read_text()
        self.feature_config = ast.literal_eval(feature_config)
        
        # get importance feature
        self.important_features = (self.work_dir/"model"/model_name/"feature_columns.txt").read_text().split("\n")

        # refenence data
        self.reference_data = pd.read_parquet(str(self.work_dir/"train"/"raw_train.parquet"), engine="pyarrow")[self.important_features]
        self.reference_data = {column: self.reference_data[column].values.tolist() for column in self.reference_data.columns if column in self.important_features}

    def __call__(self, columns: List[str], rows: List[List[Any]]) -> Tuple[bool, NDArray]:
        current_data = {column: [r[i] for r in rows] for i, column in enumerate(columns) if column in self.important_features}

        is_drift = self.stattest.detect_drift_data(
            reference_data=self.reference_data,
            current_data=current_data,
            feature_config=self.feature_config
        )
        
        X = Pool(
            data=[current_data[column] for column in self.important_features],
            cat_features=[i for i, column in enumerate(self.important_features) if column in self.feature_config['category_columns']]
        )

        predictions = self.model.predict(
            X,
            thread_count=3,
            verbose=False,
            task_type='CPU'
        )

        predictions = np.squeeze(predictions)
        return is_drift, list(predictions)
    
