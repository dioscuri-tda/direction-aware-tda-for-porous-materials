import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from src.data import MDDataset, DescDimReducer, split_into_folds_nostratification, get_npy_shape, split_into_folds_stratification
from src.generate_model import generate_model, ALLOWED_MODELS
from src.inputoutput import get_experiment_name, save_to_pickle, save_to_json
from src.training import get_optimizer
from src.utils import DownSample, EarlyStopper, RandomRoll, RandomFlip
from src.validation import validate_one_epoch

TARGET_GROUPS = {
    "uniaxial": [
        "C_xx_xx", "C_yy_yy", "C_zz_zz",
    ],
    "poisson": [
        "C_xx_yy", "C_xx_zz", "C_yy_zz",
    ],
    "shear": [
        "C_yz_yz", "C_xz_xz", "C_xy_xy",
    ],
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


def resolve_targets(targets_arg: list, df_cols: list) -> list:
    if len(targets_arg) == 1 and targets_arg[0] in TARGET_GROUPS:
        keyword = targets_arg[0]
        cols = [c for c in TARGET_GROUPS[keyword] if c in df_cols]
        if not cols:
            raise ValueError(
                f"Target group '{keyword}' requested but no matching columns found in database."
            )
        return cols
    missing = [t for t in targets_arg if t not in df_cols]
    if missing:
        raise ValueError(f"Target column(s) not found in database: {missing}")
    return targets_arg


def check_npy_cubic_and_equalsize(df: pd.DataFrame, path_column: str = "npy_path") -> int:
    required_npy_shape = None
    for row_id, row in tqdm(df.iterrows(), total=len(df), desc="Checking dataset"):
        npy_path = row[path_column]
        npy_shapes = get_npy_shape(npy_path)
        assert (
            npy_shapes[0] == npy_shapes[1] == npy_shapes[2]
        ), f"Structures must be cubic. Non cubic structure found in {npy_path}"
        if row_id == 0:
            required_npy_shape = npy_shapes[0]
        else:
            assert (
                required_npy_shape == npy_shapes[0]
            ), f"Structure of different resolutions found. Expected {required_npy_shape} but {npy_shapes[0]} found for {npy_path}"
    return required_npy_shape


def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=str, default="md_database.csv", help="Path to database .csv file")
    parser.add_argument("--outputdir", type=str, default="CNN/results", help="Path where results are stored")
    parser.add_argument("--model_name", choices=ALLOWED_MODELS, default=ALLOWED_MODELS[0], help="Model type")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--size", type=int, default=80)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--optimizer", choices=["SGD", "Adam"], default="Adam", help="Which optimier to use")
    parser.add_argument("--aug_roll_ratio", type=float, default=0.0, help="RandomRoll augmentation")
    parser.add_argument("--suffix", type=str, default="", help="Suffix appended to filename")
    parser.add_argument("--use_descriptors", action="store_true")
    parser.add_argument("--no_structure", action="store_true")
    parser.add_argument("--fc_first_size", nargs="+", type=int, default=[33, 33, 21])
    parser.add_argument("--fc_second_size", nargs="+", type=int, default=[32, 32, 32])
    parser.add_argument("--p_dropout", type=float, default=0.3, help="Dropout for fullconnected")
    parser.add_argument("--cv_folds", type=int, default=5, help="Number of folds")
    parser.add_argument("--folds", nargs="+", type=int, default=[0], help="Which folds to use")
    parser.add_argument("--p_flip", type=float, default=0.0, help="Flip probability")
    parser.add_argument("--gradient_clip", type=float, default=0.0, help="Gradient clip value")
    parser.add_argument("--remove_model", action="store_true")
    parser.add_argument("--use_stratification", action="store_true")
    parser.add_argument(
        "--targets", nargs="+", type=str, default=["full_tensor"],
        help=(
            "Target column(s) to predict.  Predefined subsets of increasing size:\n"
            "  uniaxial            — C_xx_xx, C_yy_yy, C_zz_zz (3)\n"
            "  poisson             — C_xx_yy, C_xx_zz, C_yy_zz (3)\n"
            "  shear               — C_yz_yz, C_xz_xz, C_xy_xy (3)\n"
            "  offdiagonal         — 12 normal-shear coupling elements\n"
            "  uniaxial+poisson    — 6 elements\n"
            "  uniaxial+poisson+shear — 9 elements\n"
            "  full_tensor         — all 21 Cij\n"
            "Or list specific columns: --targets C_xx_xx C_zz_zz"
        ),
    )
    parser.add_argument(
        "--path_column", type=str, default="npy_path",
        help="Column in the database CSV containing paths to .npy structure files.",
    )
    return vars(parser.parse_args())


def main():
    args = parse_command_line_args()
    seed = args["seed"]
    batch_size = args["batch_size"]
    n_epochs = args["n_epochs"]
    outputdir = args["outputdir"]
    model_name = args["model_name"]
    roll_ratio = args["aug_roll_ratio"]
    use_descriptors = args["use_descriptors"]
    no_structure = args["no_structure"]
    optimizer_type = args["optimizer"]
    fc_first_size = args["fc_first_size"]
    fc_second_size = args["fc_second_size"]
    p_dropout = args["p_dropout"]
    cv_folds = args["cv_folds"]
    which_folds = args["folds"]
    learning_rate = args["lr"]
    p_flip = args["p_flip"]
    target_size = args["size"]
    gradient_clip = args["gradient_clip"]
    remove_model = args["remove_model"]
    use_stratification = args["use_stratification"]
    path_column = args["path_column"]
    database = pd.read_csv(args["database"])

    target_columns = resolve_targets(args["targets"], database.columns.tolist())
    n_targets = len(target_columns)
    print(f"Predicting {n_targets} target(s): {target_columns}")

    # set seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Will run on device={device}")

    npy_size = check_npy_cubic_and_equalsize(df=database, path_column=path_column)
    print(f"Structures are cubic: {npy_size}x{npy_size}x{npy_size}")
    print(f"Structures will be rescaled to: {target_size}x{target_size}x{target_size}")

    ### shuffle the database
    database = database.sample(frac=1)

    experiment_name = get_experiment_name(args)
    experiment_path_parent = os.path.join(outputdir, experiment_name)
    os.makedirs(experiment_path_parent, exist_ok=True)
    save_to_json(
        {**args, "resolved_target_columns": target_columns, "directions": [], "features": ["cnn"]},
        os.path.join(experiment_path_parent, "arguments.json"),
    )

    downsampler = DownSample(scale_factor=target_size / npy_size)
    randomroll = RandomRoll(ratio_y=roll_ratio, ratio_z=roll_ratio)
    randomflip = RandomFlip(p_flip=p_flip)
    transform_train = transforms.Compose([transforms.ToTensor(), downsampler, randomflip, randomroll])
    transform_test = transforms.Compose([transforms.ToTensor(), downsampler])

    split_into_folds = split_into_folds_stratification if use_stratification else split_into_folds_nostratification
    split_kwargs = dict(df=database, cv_folds=cv_folds, which_folds=which_folds)
    if use_stratification:
        split_kwargs["path_column"] = path_column
    all_folds_metrics = []
    for df_train, df_val, df_test, fold_id in split_into_folds(**split_kwargs):
        print(
            f"Fold: {fold_id} Dataset size. Train={len(df_train)} Val={len(df_val)} Test={len(df_test)}  Total={len(database)}"
        )
        datadimreducer = None
        if use_descriptors:
            datadimreducer = DescDimReducer(df=df_train, target_columns=target_columns, path_column=path_column)

        trainset = MDDataset(
            df=df_train, transform=transform_train, use_descriptors=use_descriptors, descriptor_transform=datadimreducer,
            target_columns=target_columns, path_column=path_column,
        )
        valset = MDDataset(
            df=df_val, transform=transform_test, use_descriptors=use_descriptors, descriptor_transform=datadimreducer,
            target_columns=target_columns, path_column=path_column,
        )
        testset = MDDataset(
            df=df_test, transform=transform_test, use_descriptors=use_descriptors, descriptor_transform=datadimreducer,
            target_columns=target_columns, path_column=path_column,
        )
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)
        valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

        model = generate_model(
            model_name=model_name,
            use_descriptors=use_descriptors,
            no_structure=no_structure,
            fc_first_size=fc_first_size,
            fc_second_size=fc_second_size,
            p_dropout=p_dropout,
            n_targets=n_targets,
        )
        model.to(device)

        loss_fn = torch.nn.MSELoss(reduction="sum")
        optimizer = get_optimizer(model=model, optimizer_type=optimizer_type, lr=learning_rate)
        experiment_path = os.path.join(experiment_path_parent, f"fold_{fold_id}")
        model_path = os.path.join(experiment_path, "best_model.pth")
        os.makedirs(experiment_path, exist_ok=True)
        save_to_json(args, os.path.join(experiment_path, "parameters.json"))

        history = []
        writer = SummaryWriter(os.path.join(experiment_path, "tensorboard"))
        early_stopper = EarlyStopper(patience=10, model_path=model_path)
        for epoch in range(n_epochs):
            running_loss = 0
            running_count = 0
            for data in tqdm(trainloader, desc="Train"):
                inputs = data[0].to(device)
                targets = data[1].to(device)
                if use_descriptors:
                    descriptors = data[3].to(device)
                    y_pred = model(inputs, descriptors)
                else:
                    y_pred = model(inputs)
                # forward, backward, and then weight update
                loss = loss_fn(y_pred, targets)
                optimizer.zero_grad()
                loss.backward()
                if gradient_clip > 1e-6:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
                optimizer.step()
                running_loss += loss.item()
                running_count += len(data)
            totalloss = running_loss / running_count

            trainloss = validate_one_epoch(
                model=model,
                dataloader=trainloader,
                loss_function=loss_fn,
                device=device,
                use_descriptors=use_descriptors,
                writer=writer,
                dataloadertype="train",
                epoch=epoch,
                target_columns=target_columns,
            )
            valloss = validate_one_epoch(
                model=model,
                dataloader=valloader,
                loss_function=loss_fn,
                device=device,
                use_descriptors=use_descriptors,
                writer=writer,
                dataloadertype="val",
                epoch=epoch,
                target_columns=target_columns,
            )
            testloss = validate_one_epoch(
                model=model,
                dataloader=testloader,
                loss_function=loss_fn,
                device=device,
                use_descriptors=use_descriptors,
                writer=writer,
                dataloadertype="test",
                epoch=epoch,
                target_columns=target_columns,
            )
            history.append({"epoch": epoch, "trainloss": trainloss, "valloss": valloss, "testloss": testloss})
            print(
                f'Fold {fold_id} Ep {epoch}: train runningloss={totalloss} loss={trainloss["loss"]:.4f} mse={trainloss["mse"]:.4f} r2={trainloss["r2"]:.4f} val loss={valloss["loss"]:.4f} r2={valloss["r2"]:.4f} bestloss: {early_stopper.min_validation_loss:4f} counter={early_stopper.counter} test loss={testloss["loss"]:.4f} r2={testloss["r2"]:.4f}'
            )
            if early_stopper.early_stop(valloss["loss"], model=model):
                print("Break due to early stopping")
                break
        save_to_pickle(history, os.path.join(experiment_path, "history.pickle"))
        save_to_json(trainloss, os.path.join(experiment_path, "summary_train.json"))
        save_to_json(valloss, os.path.join(experiment_path, "summary_val.json"))
        save_to_json(testloss, os.path.join(experiment_path, "summary_test.json"))

        print(f"Restoring best model from: {model_path}")
        model.load_state_dict(torch.load(model_path))
        trainloss = validate_one_epoch(
            model=model,
            dataloader=trainloader,
            loss_function=loss_fn,
            device=device,
            use_descriptors=use_descriptors,
            writer=None,
            dataloadertype="train",
            epoch=epoch,
            target_columns=target_columns,
        )
        valloss = validate_one_epoch(
            model=model,
            dataloader=valloader,
            loss_function=loss_fn,
            device=device,
            use_descriptors=use_descriptors,
            writer=None,
            dataloadertype="val",
            epoch=epoch,
            target_columns=target_columns,
        )
        testloss = validate_one_epoch(
            model=model,
            dataloader=testloader,
            loss_function=loss_fn,
            device=device,
            use_descriptors=use_descriptors,
            writer=None,
            dataloadertype="test",
            epoch=epoch,
            target_columns=target_columns,
        )
        all_folds_metrics.append(testloss)
        save_to_json(trainloss, os.path.join(experiment_path, "summary_train_best_model.json"))
        save_to_json(valloss, os.path.join(experiment_path, "summary_val_best_model.json"))
        save_to_json(testloss, os.path.join(experiment_path, "summary_test_best_model.json"))

        if remove_model:
            os.remove(os.path.join(experiment_path, "best_model.pth"))
    # after all folds are done
    per_target_keys = ["r2_per_target", "mse_per_target", "mae_per_target", "mape_per_target"]
    all_folds_df = pd.DataFrame(all_folds_metrics)
    avg_metrics = all_folds_df.select_dtypes(include="number").mean().to_dict()
    for key in per_target_keys:
        if key in all_folds_df.columns:
            col_dicts = all_folds_df[key].tolist()
            avg_metrics[key] = {
                col: float(np.mean([d[col] for d in col_dicts]))
                for col in col_dicts[0]
            }
    # catboost-compatible format: top-level r2/mae/mape + per_target sub-dicts
    catboost_avg = {
        "r2":   avg_metrics.get("r2",   float("nan")),
        "mae":  avg_metrics.get("mae",  float("nan")),
        "mape": avg_metrics.get("mape", float("nan")),
    }
    if "r2_per_target" in avg_metrics:
        catboost_avg["per_target"] = {
            col: {
                "r2":   avg_metrics["r2_per_target"].get(col,   float("nan")),
                "mae":  avg_metrics.get("mae_per_target", {}).get(col, float("nan")),
                "mape": avg_metrics.get("mape_per_target", {}).get(col, float("nan")),
            }
            for col in avg_metrics["r2_per_target"]
        }
    save_to_json(catboost_avg, os.path.join(experiment_path_parent, "metrics_fold_avg.json"))



if __name__ == "__main__":
    main()
