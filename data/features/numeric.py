import pandas as pd
from dataclasses import dataclass
from typing import Mapping, List

@dataclass
class Distribution:
    mean: float
    std: float

class NumericTransformer:
    def __init__(self):
        self.columns: List[str]=[]
        self.distribution: Mapping[str, Distribution]={}

    def transform(self, df: pd.DataFrame, column: str, **kwargs) -> pd.DataFrame:
        # get distribution
        distribution: Distribution=self.distribution[column]
        
        # do transform
        df[column] = df[column].apply(lambda x: (x-distribution.mean)/distribution.std)
        
        return df
    
    def get_numeric_distribution(self, df: pd.DataFrame, column: str):
        mean = df[column].mean()
        std  = df[column].std()

        self.distribution[column] = Distribution(
            mean=mean,
            std=std
        )

        if column not in self.columns:
            self.columns.append(column)

    

