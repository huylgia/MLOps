import pandas
from typing import List, Mapping, Tuple

def build_category_map(df: pandas.DataFrame, category_columns: List[str]) -> Tuple[pandas.DataFrame, Mapping[str, List[str]]]:
    category_map: Mapping[str, List[str]]={}

    for col in category_columns:
        # convert object to category
        df[col] = df[col].astype("category")

        # define category map
        category_map[col] = df[col].cat.categories
    
        # mapping string to index
        df[col] = df[col].cat.codes
        
    return df, category_map




