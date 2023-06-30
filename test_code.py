from data import *
import pathlib
import ast
import pandas as pd
from utils import measure_execute_time
from typing import List, Mapping, Any, Tuple

def load_reference_data(phase: str, problem: str) -> Tuple[pd.DataFrame, Mapping[str, Any]]:
    prob_dir = pathlib.Path(__file__).parent/"samples"/phase/problem

    # get category columns
    cfg_str = (prob_dir/"features_config.json").read_text()
    cfg_map = ast.literal_eval(cfg_str)
    
    # read data
    df = pd.read_parquet(str(prob_dir/"raw_train.parquet") ,engine='pyarrow')
    df, _ = build_category_map(df, cfg_map['category_columns'])

    return df, cfg_map

def load_current_data(phase: str, problem: str, cfg_map: Mapping[str, Any]) -> Mapping[str, pd.DataFrame]:
    prob_dir = pathlib.Path(__file__).parent/"store_data"/phase/problem

    current_data_map: Mapping[str, pd.DataFrame]={}
    for current_file in prob_dir.glob("*.csv"):
        request_id = current_file.name.split("_")[0]

        df = pd.read_csv(current_file)
        df, _ = build_category_map(df, cfg_map['category_columns'])

        current_data_map[request_id] = df

    return current_data_map

def run(phase: str, problem: str):
    reference_data, cfg_map = load_reference_data(phase, problem)
    current_data_map = load_current_data(phase, problem, cfg_map)

    for request_id, current_data in current_data_map.items():
        args = dict(
            reference=reference_data.drop(columns=["label"]),
            current=current_data
        )
        execute_time = measure_execute_time(check_data_drift, args, 100)

        is_drift, report = check_data_drift(
            **args
        )

        print(f"{request_id}: {is_drift}\t{execute_time} seconds")
        
        break
    
if __name__ == "__main__":
    # run("phase-1", "prob-1")