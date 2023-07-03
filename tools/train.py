import pathlib, sys
__dir__ = pathlib.Path(__file__).parent
sys.path.append(str(__dir__.parent))

import numpy
import pickle

from data import DataLoader
from model import Trainer

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

    # train
    trainer = Trainer(train_work_dir, args=args)
    trainer.train(X_train, Y_train, X_val, Y_val)

    # save
    f = (train_work_dir/"model.pickle").open(mode="wb")
    pickle.dump(trainer.model, f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    run("phase-2", "prob-1")
    run("phase-2", "prob-2")