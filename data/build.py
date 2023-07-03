import ast
import pandas as pd
from typing import Dict, Tuple, List, Any
from pathlib import Path
from sklearn.model_selection import train_test_split
from numpy.typing import NDArray
from .features import CategoryTransformer, NumericTransformer, save_tranformer

class DataLoader:
    def __init__(self, work_dir: Path, data_name: str="raw_train.parquet", features_config_name: str="features_config.json") -> None:
        self.transformer = {
            "category": CategoryTransformer(),
            "numeric": NumericTransformer()
        }

        self.work_dir  = work_dir
        self.data_file = work_dir/"train"/data_name
        self.features_config_file = work_dir/"train"/features_config_name
    
    def __call__(self) -> Tuple[*NDArray]:
        self.call()

    def call(self) -> Tuple[*NDArray]:
        # get feature config
        feature_cfg_str = self.features_config_file.read_text()
        feature_cfg = ast.literal_eval(feature_cfg_str)

        # get data
        data = pd.read_parquet(str(self.data_file), engine='pyarrow')

        # transform
        data = self.transform(data=data, feature_config=feature_cfg)

        # split dataset
        X_train, X_val, Y_train, Y_val = self.split_data(data=data, target_column=feature_cfg['target_column'], test_size=0.2)

        return X_train, X_val, Y_train, Y_val

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
        
        save_tranformer(category_transformer, transformer_dir, "category_transformer")

        # tranform numeric
        numeric_transformer = NumericTransformer()
        for column in feature_config['numeric_columns']:
            numeric_transformer.get_numeric_distribution(data, column=column)
            data = numeric_transformer.transform(data, column=column)

        save_tranformer(numeric_transformer, transformer_dir, "numeric_transformer")

        return data
    
    def split_data(self, data: pd.DataFrame, target_column: str, test_size: float=0.2) -> Tuple[*NDArray]:
        data = data.sample(frac=1, random_state=42)
        
        # get X, Y
        Y = data[target_column].values
        X = data.drop(columns=[target_column]).values

        return train_test_split(X, Y, test_size=test_size, random_state=42)    