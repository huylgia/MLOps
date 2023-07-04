import pandas
from typing import List, Mapping

class CategoryTransformer:
    def __init__(self):
        self.columns: List[str]=[]
        self.mapping: Mapping[str, List[str]]={}

    def transform(self, df: pandas.DataFrame, column: str, is_onehot: bool=False):
        # get category map 
        category_map = self.mapping[column]
        
        # tranform category into index
        df[column] = df[column].apply(lambda x: category_map.index(x) if x in category_map else -1)

        # onehot encoding
        if is_onehot:
            onehot = pandas.get_dummies(df[column], prefix=column)
            onehot = onehot.astype(int)

            # drop category column and add onehot
            df = df.drop(columns=[column])
            df = pandas.concat([df, onehot], axis=1)

        return df
    
    def get_category_index(self, df: pandas.DataFrame, column: str):
        # convert object to category
        df[column] = df[column].astype("category")

        # get category index
        self.mapping[column] = df[column].cat.categories.tolist()

        if column not in self.columns:
            self.columns.append(column)




