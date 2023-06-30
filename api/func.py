from datetime import datetime
import pathlib
import pandas as pd
from typing import List
from dataclasses import dataclass

__DIR__ = pathlib.Path(__file__).parent

@dataclass
class Output:
    id: str
    predictions: List[float]
    drift: int


async def store_data(id: str, columns: List[str], rows: List[List[float]], phase: str="phase-1", problem: str="prob-1") -> Output:
    current = datetime.now()
    
    # create store dir
    store_dir = __DIR__.parent/"store_data"/phase/problem/f"{current.month}_{current.day}_{current.hour}_{current.minute}"
    store_dir.mkdir(exist_ok=True, parents=True)

    # file to store data
    csv_file = f"{store_dir}/{id}.csv"
    
    # create data
    data = pd.DataFrame(rows, columns=columns)
    data.to_csv(csv_file, index=None)

    # output to return
    output = Output(
        id=id,
        predictions=[1.0 for _ in range(len(rows))],
        drift=0
    )

    return output

async def predict(id: str, columns: List[str], rows: List[List[float]]) -> Output:
    output = Output(
        id=id,
        predictions=[1.0 for _ in range(len(rows))],
        drift=[0 for _ in range(len(rows))]
    )

    return output

