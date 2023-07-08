import pathlib, sys
__dir__ = pathlib.Path(__file__).parent
sys.path.append(str(__dir__.parent))

import numpy

from data import DataLoader
from model import Trainer, save_weight

def run(phase: str, problem: str, model_name: str="CatBoostClassifier", postfix:str="", data_name="raw_train.parquet"):
    prob_dir = __dir__.parent/"samples"/phase/problem

    # dataset
    loader = DataLoader(prob_dir, data_name=data_name, postfix=postfix)
    X, Y = loader.call(numeric_is_trans=True, caterogy_is_trans=True)

    # model
    train_work_dir = (prob_dir/"model"/f"{model_name}{postfix}")
    train_work_dir.mkdir(parents=True, exist_ok=True)
    args = {
        "loss_function": 'MultiClass' if len(numpy.unique(Y)) > 2 else "Logloss",
        'eval_metric': "TotalF1" if len(numpy.unique(Y)) > 2 else "F1",
        "train_dir": train_work_dir
    }

    trainer = Trainer(train_work_dir, args=args)
    # ======================= Train full features =======================
    trainer.initialize()
    trainer.train(X, Y, cv=5, train_size=0.8)
    
    # get feature important
    importance_scores = trainer.model.feature_importances_
    importance_feature_indexs = numpy.where(importance_scores >= 1)[0]

    # ======================= Train important features =======================
    trainer.initialize()
    trainer.train(
        X[:, importance_feature_indexs], 
        Y, 
        cv=5, 
        train_size=0.8
    )

    # ======================= Save weight =======================
    save_weight(trainer.model, train_work_dir, f"model")
    
    importance_columns = loader.data.columns[importance_feature_indexs]
    (train_work_dir/f"feature_columns.txt").write_text("\n".join(importance_columns))

if __name__ == "__main__":
    run("phase-2", "prob-1", postfix="_2", data_name="raw_train_2.parquet")
    run("phase-2", "prob-2", postfix="_2", data_name="raw_train_2.parquet")