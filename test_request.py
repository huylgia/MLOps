import requests
import pandas as pd

def main():
    data = pd.read_parquet(
        "samples/phase-2/prob-1/train/raw_train.parquet",
        engine='pyarrow'
    )
    label = data['label'].values
    data.drop(columns=["label"], inplace=True)

    requests.post(
        "http://localhost:1234/phase-2/prob-1/predict",
        json={
            "id": "0",
            "columns": data.columns.tolist(),
            "rows": data.values[:10].tolist()
        }
    )

main()