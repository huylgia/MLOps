from catboost import Pool, CatBoostClassifier
from typing import Dict
from pathlib import Path

from .default import CONFIG

class Trainer:
    def __init__(self, work_dir: Path, model_name: str="CatBoostClassifier", args: Dict={}) -> None:
        self.work_dir  = work_dir
        self.model_name = model_name
        self.args = args

        self.args.update(CONFIG[model_name])
        
    def initialize(self):
        self.model: CatBoostClassifier = eval(self.model_name)(**self.args)

    def train(self, X_train, Y_train, X_val, Y_val):
        train_dataset = Pool(
            data=X_train,
            label=Y_train,
        )

        val_dataset = Pool(
            data=X_val,
            label=Y_val,
        )

        self.model.fit(
            train_dataset,
            eval_set=val_dataset,
            verbose=False
        )    