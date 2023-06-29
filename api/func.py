import time
import pathlib
import pandas as pd
from typing import List
from dataclasses import dataclass

__DIR__ = pathlib.Path(__file__).parent

@dataclass
class Output:
    id: int
    predictions: List[float]
    drift: int


async def store_data(id: int, columns: List[str], rows: List[List[float]], phase: str="phase-1", problem: str="prob-1") -> Output:
    store_dir = __DIR__.parent/"store_data"/phase/problem
    store_dir.mkdir(exist_ok=True, parents=True)

    # file to store data
    csv_file = f"{store_dir}/{id}_{int(time.time())}.csv"
    
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

async def predict(id: int, columns: List[str], rows: List[List[float]]) -> Output:
    output = Output(
        id=id,
        predictions=[1.0 for _ in range(len(rows))],
        drift=[0 for _ in range(len(rows))]
    )

    return output

