import ast
import pandas as pd
from typing import Dict, Tuple, List, Any, Iterable
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from numpy.typing import NDArray
from .features import CategoryTransformer, NumericTransformer, save_tranformer
from .drift import StatTest

class DataLoader:
    def __init__(self, work_dir: Path, postfix: str, data_name: str="raw_train.parquet", features_config_name: str="features_config.json") -> None:
        self.postfix = postfix

        # create drift detector
        self.stattest = StatTest()

        # create transformer
        self.transformer = {
            "category": CategoryTransformer(),
            "numeric": NumericTransformer()
        }

        # get some info from dataset
        self.work_dir  = work_dir
        self.data_file = work_dir/"train"/data_name
        self.features_config_file = work_dir/"train"/features_config_name

        # get feature config
        feature_cfg_str = self.features_config_file.read_text()
        self.feature_config = ast.literal_eval(feature_cfg_str)

        # get data
        self.data = pd.read_parquet(str(self.data_file), engine='pyarrow')

    def __call__(self) -> Tuple[*NDArray]:
        self.call()

    def call(self) -> Tuple[*NDArray]:        
        # transform
        data = self.transform(data=self.data, feature_config=self.feature_config)
        
        kfold = self.split_data(data, self.feature_config['target_column'], 0.2, 42)

        return kfold
    
    def transform(self, data: pd.DataFrame, feature_config: Dict):
        transformer_dir = self.work_dir/"transformer"
        transformer_dir.mkdir(exist_ok=True)

        # add target column into category column if column type is string
        if isinstance(data[feature_config['target_column']].dtype, str):
            feature_config['category_columns'].append(feature_config['target_column'])

        # tranform category
        category_transformer = CategoryTransformer()
        for column in feature_config['category_columns']:
            category_transformer.get_category_index(data, column=column)
            
            data = category_transformer.transform(
                data, 
                column=column, 
                is_onehot=False
            )
        
        save_tranformer(category_transformer, transformer_dir, f"category_transformer_{self.postfix}")

        # tranform numeric
        numeric_transformer = NumericTransformer()
        for column in feature_config['numeric_columns']:
            numeric_transformer.get_numeric_distribution(data, column=column)
            data = numeric_transformer.transform(data, column=column)

        save_tranformer(numeric_transformer, transformer_dir, f"numeric_transformer_{self.postfix}")

        return data
    
    def split_data(self, data: pd.DataFrame, target_column: str, test_size: float=0.2, random_state=42) -> Iterable[*NDArray]:
        data = data.sample(frac=1, random_state=random_state)
        print(data.shape)
        # get X, Y
        Y = data[target_column].values
        X = data.drop(columns=[target_column]).values

        sss = StratifiedShuffleSplit(n_splits=5, test_size=test_size, random_state=random_state)
        for (train_index, test_index) in sss.split(X, Y):
            yield (X[train_index], X[test_index], Y[train_index], Y[test_index])
