from catboost import Pool, CatBoostClassifier
from typing import Dict
from pathlib import Path

from .default import CONFIG

class Trainer:
    def __init__(self, work_dir: Path, model_name: str="CatBoostClassifier", args: Dict={}) -> None:
        self.work_dir  = work_dir
        self.model_name = model_name

        self.args = CONFIG[model_name]
        self.args.update(args)
        
        self.grid = CONFIG['grid']

    def initialize(self):
        self.model: CatBoostClassifier = eval(self.model_name)(**self.args)

    def train(self, X, Y, cv: int=5, train_size: float=0.8):
        self.model.grid_search(
            self.grid,
            cv=5,
            train_size=0.8,
            stratified=True,
            shuffle=True,
            X=X,
            y=Y,
        )            