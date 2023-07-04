import pathlib, sys
__dir__ = pathlib.Path(__file__).parent
sys.path.append(str(__dir__.parent))

import numpy
import pickle

from data import DataLoader
from model import Trainer, save_weight

def run(phase: str, problem: str, model_name: str="CatBoostClassifier"):
    prob_dir = __dir__.parent/"samples"/phase/problem

    # dataset
    loader = DataLoader(prob_dir)
    X_train, X_val, Y_train, Y_val = loader.call()

    # model
    train_work_dir = (prob_dir/"model"/model_name)
    train_work_dir.mkdir(parents=True, exist_ok=True)
    args = {
        "loss_function": 'MultiClass' if len(numpy.unique(Y_train)) > 2 else "Logloss",
        "train_dir": train_work_dir
    }

    trainer = Trainer(train_work_dir, args=args)
    # ======================= Train full features =======================
    trainer.initialize()
    trainer.train(X_train, Y_train, X_val, Y_val)
    
    # get feature important
    importance_scores = trainer.model.feature_importances_
    importance_feature_indexs = numpy.where(importance_scores >= 1)[0]

    # ======================= Train important features =======================
    trainer.initialize()
    trainer.train(
        X_train[:, importance_feature_indexs], 
        Y_train, 
        X_val[:, importance_feature_indexs], 
        Y_val
    )

    # ======================= Save weight =======================
    save_weight(trainer.model, train_work_dir, "model")
    
    importance_columns = loader.data.columns[importance_feature_indexs]
    (train_work_dir/"feature_columns.txt").write_text("\n".join(importance_columns))

if __name__ == "__main__":
    run("phase-2", "prob-1")
    run("phase-2", "prob-2")