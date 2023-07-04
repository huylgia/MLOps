import pickle
from pathlib import Path

from catboost import CatBoostClassifier

def save_weight(obj: CatBoostClassifier, store_dir: Path, name: str):
    f = (store_dir/f"{name}.pickle").open(mode="wb")
    pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_weight(store_dir: Path, name: str) -> CatBoostClassifier:
    f = (store_dir/f"{name}.pickle").open(mode="rb")

    return pickle.load(f)