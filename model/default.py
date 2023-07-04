CONFIG = {
    "CatBoostClassifier":
        {   
            "use_best_model": True,
            "iterations": 500, 
            "learning_rate": 0.05,
            "depth": 7,
            "rsm": 1.0,
            "eval_metric": "AUC",
            "task_type": "GPU",
            "devices": '0:1'
        }
}