from datetime import datetime
import pathlib, sys
__DIR__ = pathlib.Path(__file__).parent
sys.path.append(__DIR__.parent)

import pandas as pd
from typing import List, Any
from dataclasses import dataclass

import numpy as np
from tools.predict import Predictor
import json

@dataclass
class Output:
    id: str
    predictions: List[Any]
    drift: int

def store_data(id: str, columns: List[str], rows: List[List[Any]], phase: str="phase-1", problem: str="prob-1"):
    current = datetime.now()
    
    # create store dir
    store_dir = __DIR__.parent/"samples"/phase/problem/"test"/f"{current.month}_{current.day}_{current.hour}_{current.minute}"
    store_dir.mkdir(exist_ok=True, parents=True)

    # file to store data
    json_file = store_dir/f"{id}.json"
    data = {
        'id': id,
        'columns': columns,
        'rows': rows,
    }
    
    obj = json.dumps(data, indent=4)
    json_file.write_text(obj)
    
async def predict(id: str, columns: List[str], rows: List[List[Any]], predictor: Predictor) -> Output:
    data = pd.DataFrame(rows, columns=columns)

    # output to return
    is_drift, predictions = predictor(data)    
    output = Output(
        id=id,
        predictions=np.squeeze(predictions).tolist(),
        drift=int(is_drift)
    )

    return output


