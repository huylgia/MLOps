import ast
import pandas as pd
from typing import Tuple
from pathlib import Path
from sklearn.model_selection import train_test_split
from numpy.typing import NDArray

class DataLoader:
    def __init__(self, work_dir: Path, data_name: str="raw_train.parquet", features_config_name: str="features_config.json") -> None:
        # get some info from dataset
        self.work_dir  = work_dir
        self.data_file = work_dir/"train"/data_name
        self.features_config_file = work_dir/"train"/features_config_name

        # get feature config
        feature_cfg_str = self.features_config_file.read_text()
        self.feature_config = ast.literal_eval(feature_cfg_str)

        # get data
        data = pd.read_parquet(str(self.data_file), engine='pyarrow')

        self.data    = data.sample(frac=1, random_state=42)
        self.columns = data.columns.to_list()

    def __call__(self) -> Tuple[*NDArray]:
        self.call()

    def call(self) -> Tuple[*NDArray]:       
        Y = self.data[self.feature_config['target_column']].values
        X = self.data.drop(columns=[self.feature_config['target_column']]).values

        return train_test_split(X, Y, test_size=0.2, random_state=42)