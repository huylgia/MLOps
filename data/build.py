import ast
import pandas as pd
from typing import Dict, Tuple, List, Any, Iterable
from pathlib import Path
from numpy.typing import NDArray
from .features import CategoryTransformer, NumericTransformer, save_tranformer
from .drift import StatTest
from .utils import handle_duplicate

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
        self.data = handle_duplicate(self.data)
        
    def __call__(self, **args) -> Tuple[*NDArray]:
        self.call(**args)

    def call(self, numeric_is_trans: bool=True, caterogy_is_trans: bool=True) -> Tuple[*NDArray]:        
        # transform
        data = self.transform(data=self.data, feature_config=self.feature_config, numeric_is_trans=numeric_is_trans, caterogy_is_trans=caterogy_is_trans)
        data = data.sample(frac=1, random_state=42)
        
        # get X, Y
        Y = data[self.feature_config['target_column']].values
        X = data.drop(columns=[self.feature_config['target_column']]).values
        
        return X, Y
    
    def transform(self, data: pd.DataFrame, feature_config: Dict, numeric_is_trans: bool=True, caterogy_is_trans: bool=True):
        transformer_dir = self.work_dir/"transformer"
        transformer_dir.mkdir(exist_ok=True)

        # add target column into category column if column type is string
        if isinstance(data[feature_config['target_column']].dtype, str):
            feature_config['category_columns'].append(feature_config['target_column'])

        # tranform category
        if caterogy_is_trans:
            category_transformer = CategoryTransformer()
            for column in feature_config['category_columns']:
                category_transformer.get_category_index(data, column=column)
                
                data = category_transformer.transform(
                    data, 
                    column=column, 
                    is_onehot=False
                )
        
            save_tranformer(category_transformer, transformer_dir, f"category_transformer{self.postfix}")

        # tranform numeric
        if numeric_is_trans:
            numeric_transformer = NumericTransformer()
            for column in feature_config['numeric_columns']:
                numeric_transformer.get_numeric_distribution(data, column=column)
                data = numeric_transformer.transform(data, column=column)

            save_tranformer(numeric_transformer, transformer_dir, f"numeric_transformer{self.postfix}")

        return data