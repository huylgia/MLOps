from data import *
import pathlib
import ast
import pandas as pd
from utils import measure_execute_time
from typing import List, Mapping, Any, Tuple
from sklearn.model_selection import train_test_split

def load_reference_data(phase: str, problem: str) -> Tuple[pd.DataFrame, Mapping[str, Any]]:
    prob_dir = pathlib.Path(__file__).parent/"samples"/phase/problem

    # get category columns
    feature_cfg_str = (prob_dir/"features_config.json").read_text()
    feature_cfg = ast.literal_eval(feature_cfg_str)
    
    # read data
    df = pd.read_parquet(str(prob_dir/"raw_train.parquet") ,engine='pyarrow')

    # tranform category
    category_transformer = CategoryTransformer()
    for column in feature_cfg['category_columns']:
        category_transformer.get_category_index(df, column=column)
        df = category_transformer.transform(df, column=column, is_onehot=True)
    
    save_tranformer(category_transformer, prob_dir, "category_transformer")

    # tranform numeric
    numeric_transformer = NumericTransformer()
    for column in feature_cfg['numeric_columns']:
        numeric_transformer.get_numeric_distribution(df, column=column)
        df = numeric_transformer.transform(df, column=column)

    save_tranformer(numeric_transformer, prob_dir, "numeric_transformer")

def load_current_data(phase: str, problem: str) -> Mapping[str, pd.DataFrame]:
    reference_prob_dir = pathlib.Path(__file__).parent/"samples"/phase/problem
    current_prob_dir = pathlib.Path(__file__).parent/"store_data"/phase/problem

    # load tranformer
    category_transformer = load_tranformer(reference_prob_dir, "category_transformer")
    numeric_transformer = load_tranformer(reference_prob_dir, "numeric_transformer")

    # transform for current data
    current_data_map: Mapping[str, pd.DataFrame]={}
    for current_file in current_prob_dir.glob("*.csv"):
        request_id = current_file.name.split("_")[0]

        # read df
        df = pd.read_csv(current_file)

        # do transform
        for column in df.columns:
            if column in category_transformer.columns:
                transformer = category_transformer
            elif column in numeric_transformer.columns:
                transformer = numeric_transformer
            else:
                transformer = None

            if transformer is not None:
                df = transformer.transform(df, column=column, is_onehot=False)
            else:
                df = df.drop(columns=column)

        current_data_map[request_id] = df

    return current_data_map

load_reference_data("phase-1","prob-1")
load_current_data("phase-1","prob-1")