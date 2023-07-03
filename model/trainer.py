from catboost import Pool, CatBoostClassifier
from typing import Dict
from pathlib import Path
from .default import CONFIG

class Trainer:
    def __init__(self, work_dir: Path, model_name: str="CatBoostClassifier", args: Dict={}) -> None:
        self.work_dir  = work_dir

        args.update(CONFIG[model_name])
        self.model: CatBoostClassifier = eval(model_name)(**args)

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
            verbose=True
        )



    
    