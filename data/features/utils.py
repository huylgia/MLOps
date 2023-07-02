import pickle
from pathlib import Path
from typing import Union
from .numeric import NumericTransformer
from .category import CategoryTransformer

def save_tranformer(obj: Union[NumericTransformer, CategoryTransformer], store_dir: Path, name: str):
    f = (store_dir/f"{name}.pickle").open(mode="wb")
    pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_tranformer(store_dir: Path, name: str) -> Union[NumericTransformer, CategoryTransformer]:
    f = (store_dir/f"{name}.pickle").open(mode="rb")

    return pickle.load(f)
