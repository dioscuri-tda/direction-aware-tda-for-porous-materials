"""
Train CatBoost on directional descriptors to predict elastic tensor components.

Extends train_catboost.py with:
  --targets     one or more target columns, or a predefined subset keyword
  --directions  subset of direction tags to use (e.g. 1_0_0 or compact 100);
                if omitted, all matching columns are used
  --features    descriptor types matched by column substring (por, tpc, ecp, …)
  --path_col    name of the column holding structure paths (default: path)
  --database    one or more CSV files; if multiple are given they are inner-joined
                on path_col and their feature columns are combined

An arguments.json is written to the experiment root with all CLI args plus
the resolved feature and target column lists — useful for later analysis.
"""

import argparse
import datetime
import os
import random

import catboost as cb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

from src.data import split_into_folds_nostratification, split_into_folds_stratification
from src.inputoutput import save_to_json


TARGET_GROUPS = {
    # 3 diagonal normal stiffnesses — most directly tied to porosity
    "uniaxial": [
        "C_xx_xx", "C_yy_yy", "C_zz_zz",
    ],
    # 3 off-diagonal normal stiffnesses — Poisson coupling
    "poisson": [
        "C_xx_yy", "C_xx_zz", "C_yy_zz",
    ],
    # 3 diagonal shear stiffnesses
    "shear": [
        "C_yz_yz", "C_xz_xz", "C_xy_xy",
    ],
    # 12 normal-shear coupling elements — near zero for near-orthotropic structures
    "offdiagonal": [
        "C_xx_yz", "C_xx_xz", "C_xx_xy",
        "C_yy_yz", "C_yy_xz", "C_yy_xy",
        "C_zz_yz", "C_zz_xz", "C_zz_xy",
        "C_yz_xz", "C_yz_xy",
        "C_xz_xy",
    ],
}
TARGET_GROUPS["uniaxial+poisson"] = (
    TARGET_GROUPS["uniaxial"] + TARGET_GROUPS["poisson"]
)
TARGET_GROUPS["uniaxial+poisson+shear"] = (
    TARGET_GROUPS["uniaxial+poisson"] + TARGET_GROUPS["shear"]
)
TARGET_GROUPS["full_tensor"] = (
    TARGET_GROUPS["uniaxial+poisson+shear"] + TARGET_GROUPS["offdiagonal"]
)

FULL_STIFFNESS_TENSOR_COLS = TARGET_GROUPS["full_tensor"]


# ---------------------------------------------------------------------------
# Database loading
# ---------------------------------------------------------------------------

def load_and_merge_databases(paths: list[str], path_col: str) -> pd.DataFrame:
    """Load one or more CSVs and inner-join them on path_col.

    Columns already present in the merged frame (e.g. target columns that
    appear in every database) are kept from the first file and not duplicated.
    """
    dfs = [pd.read_csv(p) for p in paths]
    if len(dfs) == 1:
        return dfs[0]

    merged = dfs[0]
    for i, df in enumerate(dfs[1:], start=2):
        new_cols = [c for c in df.columns if c == path_col or c not in merged.columns]
        before = len(merged)
        merged = merged.merge(df[new_cols], on=path_col, how="inner")
        print(f"  DB {i}: {before} → {len(merged)} rows after inner join on '{path_col}'")

    return merged


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_direction_tag(s: str) -> str:
    """Normalise a direction string to underscore form.

    Accepts both compact Miller notation ('100', '111') and underscore form
    ('1_0_0', '1_1_1').  Single-digit, non-negative components are assumed for
    the compact form.
    """
    if "_" in s:
        return s
    return "_".join(list(s))   # '110' → '1_1_0'


def select_feature_columns(
    all_cols: list[str],
    features: list[str],
    directions: list[str],
    target_columns: list[str],
    path_col: str,
) -> list[str]:
    """Return columns that pass the feature-type and direction filters."""
    exclude = set(target_columns) | {path_col}
    dir_tags = [parse_direction_tag(d) for d in directions] if directions else []

    selected = []
    for col in all_cols:
        if col in exclude:
            continue
        if not any(feat in col for feat in features):
            continue
        if dir_tags and not any(f"_{t}_" in col for t in dir_tags):
            continue
        selected.append(col)
    return selected


def resolve_targets(targets_arg: list[str], df_cols: list[str]) -> list[str]:
    if len(targets_arg) == 1 and targets_arg[0] in TARGET_GROUPS:
        keyword = targets_arg[0]
        cols = [c for c in TARGET_GROUPS[keyword] if c in df_cols]
        if not cols:
            raise ValueError(f"'{keyword}' requested but no matching columns found in database.")
        return cols
    missing = [t for t in targets_arg if t not in df_cols]
    if missing:
        raise ValueError(f"Target column(s) not found in database: {missing}")
    return targets_arg


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_columns: list[str],
) -> dict:
    """Per-target and macro-averaged metrics."""
    if y_true.ndim == 1:
        y_true = y_true[:, None]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]

    per_target: dict[str, dict] = {}
    for i, col in enumerate(target_columns):
        per_target[col] = {
            "r2":   float(r2_score(y_true[:, i], y_pred[:, i])),
            "mae":  float(mean_absolute_error(y_true[:, i], y_pred[:, i])),
            "mse":  float(mean_squared_error(y_true[:, i], y_pred[:, i])),
            "mape": float(mean_absolute_percentage_error(y_true[:, i], y_pred[:, i])),
        }

    avg = {
        "r2":   float(np.mean([v["r2"]   for v in per_target.values()])),
        "mae":  float(np.mean([v["mae"]  for v in per_target.values()])),
        "mse":  float(np.mean([v["mse"]  for v in per_target.values()])),
        "mape": float(np.mean([v["mape"] for v in per_target.values()])),
    }
    return {"avg": avg, "per_target": per_target}


def get_experiment_name(args: dict) -> str:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    name = f"{timestamp}_catboost_directional"
    if args["suffix"]:
        name = f"{name}_{args['suffix']}"
    return name


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> dict:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--database", required=True, nargs="+",
                        help="One or more input CSVs. Multiple files are inner-joined "
                             "on path_col; feature columns are combined.")
    parser.add_argument("--outputdir", default="results",
                        help="Root directory for results (default: results).")
    parser.add_argument("--path_col",  default="path",
                        help="Column holding structure paths (default: path).")
    parser.add_argument("--seed",      type=int, default=100)
    parser.add_argument("--suffix",    default="",
                        help="Suffix appended to the experiment folder name.")
    parser.add_argument("--cv_folds",  type=int, default=5)
    parser.add_argument("--folds",     nargs="+", type=int, default=[0],
                        help="Which CV folds to run (default: 0).")
    parser.add_argument("--n_limit",   type=int, default=0,
                        help="Randomly subsample to this many rows (0 = no limit).")
    parser.add_argument("--use_stratification", action="store_true")
    parser.add_argument("--no_testset",         action="store_true",
                        help="Merge test split into training data.")
    parser.add_argument(
        "--targets", nargs="+", default=["C_zz_zz"],
        help=(
            "Target column(s) to predict.  Predefined subsets of increasing size:\n"
            "  uniaxial            — C_xx_xx, C_yy_yy, C_zz_zz (3)\n"
            "  poisson             — C_xx_yy, C_xx_zz, C_yy_zz (3)\n"
            "  shear               — C_yz_yz, C_xz_xz, C_xy_xy (3)\n"
            "  offdiagonal         — 12 normal-shear coupling elements\n"
            "  uniaxial+poisson    — 6 elements\n"
            "  uniaxial+poisson+shear — 9 elements\n"
            "  full_tensor         — all 21 Cij\n"
            "Or list specific columns: --targets C_xx_xx C_zz_zz E_x"
        ),
    )
    parser.add_argument(
        "--features", nargs="+", default=["por"],
        help="Descriptor type(s) matched by column-name substring "
             "(default: por).  E.g. --features por tpc ecp.",
    )
    parser.add_argument(
        "--directions", nargs="+", default=[],
        help="Direction tags to include.  Accept compact Miller notation "
             "('100', '111') or underscore form ('1_0_0', '1_1_1').  "
             "If omitted, all columns matching --features are used.",
    )
    parser.add_argument(
        "--with_porosity", action="store_true",
        help="Include the direction-independent '{feat}_porosity' column(s) "
             "alongside the directional features.  These are excluded by the "
             "normal direction filter, so this flag adds them back explicitly.",
    )
    return vars(parser.parse_args())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    np.random.seed(args["seed"])
    random.seed(args["seed"])

    if len(args["database"]) > 1:
        print(f"Merging {len(args['database'])} databases on '{args['path_col']}':")
        for p in args["database"]:
            print(f"  {p}")
    df = load_and_merge_databases(args["database"], args["path_col"])
    print(f"Dataset: {len(df)} structures, {len(df.columns)} columns")

    target_columns  = resolve_targets(args["targets"], df.columns.tolist())
    feature_columns = select_feature_columns(
        all_cols       = df.columns.tolist(),
        features       = args["features"],
        directions     = args["directions"],
        target_columns = target_columns,
        path_col       = args["path_col"],
    )

    if args["with_porosity"]:
        porosity_cols = [
            f"{feat}_porosity"
            for feat in args["features"]
            if f"{feat}_porosity" in df.columns
        ]
        missing = [
            f"{feat}_porosity"
            for feat in args["features"]
            if f"{feat}_porosity" not in df.columns
        ]
        if missing:
            print(f"  Warning: --with_porosity requested but columns not found: {missing}")
        # avoid duplicating if already selected (e.g. when --directions not specified)
        porosity_cols = [c for c in porosity_cols if c not in feature_columns]
        if porosity_cols:
            feature_columns = porosity_cols + feature_columns
            print(f"  Porosity columns added: {porosity_cols}")

    if not feature_columns:
        raise ValueError(
            "No feature columns selected.  Check --features and --directions against "
            f"the database columns: {df.columns.tolist()[:20]} …"
        )

    print(f"Targets  ({len(target_columns)}): {target_columns}")
    print(f"Features ({len(feature_columns)}): {feature_columns[:6]} {'…' if len(feature_columns) > 6 else ''}")

    # Build resolved args dict for arguments.json (saved once at experiment root)
    resolved = dict(args)
    resolved["resolved_target_columns"]  = target_columns
    resolved["resolved_feature_columns"] = feature_columns
    resolved["n_targets"]  = len(target_columns)
    resolved["n_features"] = len(feature_columns)

    keep_cols = [args["path_col"]] + target_columns + feature_columns
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    df = df.sample(frac=1, random_state=args["seed"])
    if args["n_limit"] > 0:
        df = df.sample(n=args["n_limit"], random_state=args["seed"])

    experiment_name = get_experiment_name(args)
    experiment_root = os.path.join(args["outputdir"], experiment_name)
    os.makedirs(experiment_root, exist_ok=True)
    save_to_json(resolved, os.path.join(experiment_root, "arguments.json"))

    split_fn = (
        split_into_folds_stratification
        if args["use_stratification"]
        else split_into_folds_nostratification
    )

    is_multitarget = len(target_columns) > 1
    loss_fn = "MultiRMSE" if is_multitarget else "RMSE"

    all_folds_metrics = []

    for df_train, df_val, df_test, fold_id in split_fn(
        df=df, cv_folds=args["cv_folds"], which_folds=args["folds"]
    ):
        fold_dir = os.path.join(experiment_root, f"fold_{fold_id}")
        os.makedirs(fold_dir, exist_ok=True)

        # Drop path column
        df_train = df_train.drop(columns=[args["path_col"]], errors="ignore")
        df_val   = df_val.drop(columns=[args["path_col"]], errors="ignore")
        df_test  = df_test.drop(columns=[args["path_col"]], errors="ignore")

        if args["no_testset"]:
            df_train = pd.concat([df_train, df_test])
            df_test  = df_test.iloc[0:0]

        print(
            f"Fold {fold_id}: train={len(df_train)}  val={len(df_val)}  "
            f"test={len(df_test)}  total={len(df)}"
        )

        X_train = df_train[feature_columns]
        y_train = df_train[target_columns] if is_multitarget else df_train[target_columns[0]]
        X_val   = df_val[feature_columns]
        y_val   = df_val[target_columns]   if is_multitarget else df_val[target_columns[0]]
        X_test  = df_test[feature_columns]
        y_test  = df_test[target_columns]  if is_multitarget else df_test[target_columns[0]]

        model = cb.CatBoostRegressor(loss_function=loss_fn)
        model.fit(
            X_train, y_train,
            verbose=0,
            eval_set=(X_val, y_val),
            use_best_model=True,
        )

        fold_metrics: dict = {"fold": fold_id}

        if not args["no_testset"] and len(df_test) > 0:
            y_pred    = np.array(model.predict(X_test))
            y_true_np = np.array(y_test)
            metrics = compute_metrics(y_true_np, y_pred, target_columns)
            fold_metrics.update(metrics["avg"])
            fold_metrics["per_target"] = metrics["per_target"]

            # Human-readable summary
            r2_str = f"R²={metrics['avg']['r2']:.4f}"
            if is_multitarget:
                per = metrics["per_target"]
                worst = min(per, key=lambda k: per[k]["r2"])
                best  = max(per, key=lambda k: per[k]["r2"])
                r2_str += (
                    f"  best={per[best]['r2']:.3f}({best})"
                    f"  worst={per[worst]['r2']:.3f}({worst})"
                )
            print(f"  Test  {r2_str}  MAE={metrics['avg']['mae']:.4f}")

            save_to_json(
                {
                    "avg": metrics["avg"],
                    "per_target": metrics["per_target"],
                    "target_columns": target_columns,
                    "target": y_true_np.tolist(),
                    "pred":   y_pred.tolist(),
                },
                os.path.join(fold_dir, "summary_test_best_model.json"),
            )
            all_folds_metrics.append(fold_metrics)

    if all_folds_metrics:
        folds_df = pd.DataFrame(all_folds_metrics)
        avg = folds_df.select_dtypes(include="number").mean().to_dict()

        # Average per-target metrics across folds
        if "per_target" in folds_df.columns:
            per_dicts = folds_df["per_target"].tolist()
            avg["per_target"] = {
                col: {
                    metric: float(np.mean([d[col][metric] for d in per_dicts]))
                    for metric in ("r2", "mae", "mse", "mape")
                }
                for col in target_columns
            }

        save_to_json(avg, os.path.join(experiment_root, "metrics_fold_avg.json"))
        print(f"\nAvg across folds — R²={avg.get('r2', float('nan')):.4f}  MAE={avg.get('mae', float('nan')):.4f}")

    print(f"\nResults saved to: {experiment_root}")


if __name__ == "__main__":
    main()
