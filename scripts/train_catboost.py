import argparse
import datetime
import os
import random

import catboost as cb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.metrics import r2_score

from src.data import split_into_folds_nostratification, split_into_folds_stratification
from src.inputoutput import save_to_json


def get_experiment_name(args):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    experiment_name = f"{timestamp}_catboost"
    if args["suffix"] != "":
        experiment_name = f"{experiment_name}_{args['suffix']}"
    return experiment_name


def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=str, default="md_database.csv", help="Path to database .csv file")
    parser.add_argument("--outputdir", type=str, default="results", help="Path where results are stored")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--suffix", type=str, default="", help="Suffix appended to filename")
    parser.add_argument("--cv_folds", type=int, default=5, help="Number of folds")
    parser.add_argument("--folds", nargs="+", type=int, default=[0], help="Which folds to use")
    parser.add_argument("--n_limit", type=int, default=0, help="Limit of the dataset")
    parser.add_argument("--evaluationset", type=str, default="")
    parser.add_argument("--no_testset", action="store_true")
    parser.add_argument("--use_stratification", action="store_true")
    parser.add_argument(
        "--features", nargs="+", type=str, default=["ecp", "ph_cone"],
        help="Feature sets to use (matched by column prefix). "
             "Options: ecp, ph_cone, porosity, tpc, fabric"
    )
    return vars(parser.parse_args())


def extract_target(df):
    target = df["cii"]
    df = df.drop(["cii"], axis=1)
    return df, target


def main():
    args = parse_command_line_args()
    seed = args["seed"]
    outputdir = args["outputdir"]
    cv_folds = args["cv_folds"]
    which_folds = args["folds"]
    evaluationset = args["evaluationset"]
    no_testset = args["no_testset"]
    n_limit = args["n_limit"]
    features = args["features"]
    use_stratification = args["use_stratification"]

    # set seed
    np.random.seed(seed)
    random.seed(seed)

    database = pd.read_csv(args["database"])

    assert "npy_path" in database.columns
    assert "cii" in database.columns
    assert "stress_axis" in database.columns

    columns_to_keep = ["npy_path", "cii"]
    for column in database.columns:
        for feature in features:
            if feature in column:
                columns_to_keep.append(column)
    database = database[columns_to_keep]

    print(database.columns)

    ### shuffle the database
    database = database.sample(frac=1)

    if n_limit > 0:
        database = database.sample(n=n_limit)

    experiment_name = get_experiment_name(args)
    experiment_path_parent = os.path.join(outputdir, experiment_name)

    if evaluationset:
        df_eval = pd.read_csv(evaluationset)
        eval_path = df_eval["npy_path"]
        df_eval = df_eval.drop(["npy_path"], axis=1)
        X_eval, y_eval = extract_target(df_eval)

    split_into_folds = split_into_folds_stratification if use_stratification else split_into_folds_nostratification
    all_folds_metrics = []
    for df_train, df_val, df_test, fold_id in split_into_folds(df=database, cv_folds=cv_folds, which_folds=which_folds):
        test_paths = df_test["npy_path"].to_list()
        df_test = df_test.drop(["npy_path"], axis=1)
        df_val = df_val.drop(["npy_path"], axis=1)
        df_train = df_train.drop(["npy_path"], axis=1)
        if no_testset:
            df_train = pd.concat([df_train, df_test])
            df_test = df_test.iloc[0:0]
        print(
            f"Fold: {fold_id} Dataset size. Train={len(df_train)} Val={len(df_val)} Test={len(df_test)}  Total={len(database)} shape={df_test.shape}"
        )

        experiment_path = os.path.join(experiment_path_parent, f"fold_{fold_id}")
        os.makedirs(experiment_path, exist_ok=True)
        save_to_json(args, os.path.join(experiment_path, "parameters.json"))

        X_train, y_train = extract_target(df_train)
        X_val, y_val = extract_target(df_val)
        X_test, y_test = extract_target(df_test)

        model = cb.CatBoostRegressor(loss_function="RMSE")
        model.fit(X_train, y_train, verbose=0, eval_set=(X_val, y_val), use_best_model=True)

        if not no_testset:
            test_predict = model.predict(X_test)
            metrics = {
                "database": args["database"],
                "r2": r2_score(y_test, test_predict),
                "mae": mean_absolute_error(y_test, test_predict),
                "mse": mean_squared_error(y_test, test_predict),
                "mape": mean_absolute_percentage_error(y_test, test_predict),
                "target": y_test.to_list(),
                "pred": test_predict.tolist(),
                "paths": test_paths,
            }
            all_folds_metrics.append(metrics)
            save_to_json(metrics, os.path.join(experiment_path, "summary_test_best_model.json"))

        if evaluationset:
            eval_predict = model.predict(X_eval)
            r2 = r2_score(y_eval, eval_predict)
            mae = mean_absolute_error(y_eval, eval_predict)
            metrics = {
                "r2": r2,
                "mae": mae,
                "test_paths": eval_path.tolist(),
                "target": y_eval.tolist(),
                "pred": eval_predict.tolist(),
            }
            all_folds_metrics.append(metrics)
            save_to_json(metrics, os.path.join(experiment_path, "summary_evaluation_best_model.json"))
    all_folds_metrics = pd.DataFrame(all_folds_metrics)
    avg_metrics = all_folds_metrics.select_dtypes(include="number").mean().to_dict()
    save_to_json(avg_metrics, os.path.join(experiment_path_parent, "matrics_fold_avg.json"))



if __name__ == "__main__":
    main()
