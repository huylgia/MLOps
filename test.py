import pandas, ast
from typing import Tuple, Mapping, Any
from sklearn.model_selection import train_test_split
from data import StatTest
from pathlib import Path
from utils import measure_execute_time
if __name__ == "__main__":
    df = pandas.read_parquet("samples/phase-2/prob-1/train/raw_train.parquet", engine="pyarrow")
    df.sample(frac=1, random_state=42)

    feature_cfg_str = Path("samples/phase-2/prob-1/train/features_config.json").read_text()
    feature_cfg = ast.literal_eval(feature_cfg_str)

    label = df['label'].values
    data  = df.drop(columns=['label'])

    # for col in feature_cfg['category_columns']:
    #     data[col] = data[col].astype("category")
    #     data[col] = data[col].cat.codes

    # for col in feature_cfg['numeric_columns']:
    #     df[col] -= df[col].mean()
    #     df[col] /= df[col].std()

    data = data.values

    X_train, X_val, Y_train, Y_val = train_test_split(data, label, test_size=0.2, random_state=42)
    stattest = StatTest()

    args = dict(
        reference_df=pandas.DataFrame(X_train, columns=df.columns[:-1]),
        current_df=pandas.DataFrame(X_val, columns=df.columns[:-1]), 
        feature_config=feature_cfg
    )
    a = measure_execute_time(stattest.detect_drift_data, args=args, loop=100)
    print(a, "seconds")
    # is_drift, score = stattest.detect_drift_data(

    # )