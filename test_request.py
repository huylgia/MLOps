import requests
import pandas as pd
from utils import measure_execute_time
def main():
    data = pd.read_parquet(
        "samples/phase-2/prob-2/train/raw_train.parquet",
        engine='pyarrow'
    )
    label = data['label'].values
    data.drop(columns=["label"], inplace=True)

    args = dict(
        url="http://localhost:1234/phase-2/prob-2/predict",
        json={  
            "id": "0",
            "columns": data.columns.tolist(),
            "rows": data.values[:1000].tolist()
        }
    )
    t = measure_execute_time(requests.post, args=args, loop=10)
    print(t)

main()
