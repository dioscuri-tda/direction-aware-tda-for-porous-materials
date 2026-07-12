import numpy as np
import torch
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from tqdm import tqdm


def validate_one_epoch(
    model, dataloader, loss_function, device, use_descriptors=False, writer=None, dataloadertype="train", epoch=0,
    target_columns=None,
):
    model.eval()
    if len(dataloader) == 0:
        output = {
            "loss": np.nan,
            "mse": np.nan,
            "mape": np.nan,
            "mae": np.nan,
            "r2": np.nan,
            "r2_per_target": np.nan,
            "mse_per_target": np.nan,
            "mape_per_target": np.nan,
            "mae_per_target": np.nan,
            "target": np.nan,
            "pred": np.nan,
            "paths": np.nan,
        }
        return output
    running_loss = 0
    running_count = 0
    allpredictions = []
    allpaths = []
    targets = []
    with torch.no_grad():
        for data in tqdm(dataloader, desc="Val"):
            inputs = data[0].to(device)
            target = data[1].to(device)
            if use_descriptors:
                desc = data[3].to(device)
                y_pred = model(inputs, desc)
            else:
                y_pred = model(inputs)
            allpaths.append(data[2])
            running_loss += loss_function(y_pred, target)
            running_count += len(target)
            allpredictions += y_pred.cpu().detach().tolist()
            targets += target.cpu().detach().tolist()
    model.train()
    allpredictions = np.array(allpredictions)  # [n_samples, n_targets]
    targets = np.array(targets)                # [n_samples, n_targets]
    loss = (running_loss / running_count).detach().cpu().item()
    n_targets = allpredictions.shape[1] if allpredictions.ndim == 2 else 1
    keys = target_columns if (target_columns and len(target_columns) == n_targets) else [str(i) for i in range(n_targets)]

    try:
        _r2 = r2_score(targets, allpredictions, multioutput="uniform_average")
        _r2_per = dict(zip(keys, r2_score(targets, allpredictions, multioutput="raw_values").tolist()))
    except:
        _r2 = np.NaN
        _r2_per = {k: np.NaN for k in keys}
    try:
        _mse = mean_squared_error(targets, allpredictions, multioutput="uniform_average")
        _mape = mean_absolute_percentage_error(targets, allpredictions, multioutput="uniform_average")
        _mae = mean_absolute_error(targets, allpredictions, multioutput="uniform_average")
        _mse_per = dict(zip(keys, mean_squared_error(targets, allpredictions, multioutput="raw_values").tolist()))
        _mape_per = dict(zip(keys, mean_absolute_percentage_error(targets, allpredictions, multioutput="raw_values").tolist()))
        _mae_per = dict(zip(keys, mean_absolute_error(targets, allpredictions, multioutput="raw_values").tolist()))
    except:
        _mse = np.NaN
        _mape = np.NaN
        _mae = np.NaN
        _mse_per = {k: np.NaN for k in keys}
        _mape_per = {k: np.NaN for k in keys}
        _mae_per = {k: np.NaN for k in keys}
    output = {
        "loss": loss,
        "mse": _mse,
        "mape": _mape,
        "mae": _mae,
        "r2": _r2,
        "r2_per_target": _r2_per,
        "mse_per_target": _mse_per,
        "mape_per_target": _mape_per,
        "mae_per_target": _mae_per,
        "target": targets.tolist(),
        "pred": allpredictions.tolist(),
        "paths": allpaths,
    }
    if writer:
        writer.add_scalar(f"Loss/{dataloadertype}", loss, epoch)
        writer.add_scalar(f"MAPE/{dataloadertype}", _mape, epoch)
        writer.add_scalar(f"MAE/{dataloadertype}", _mae, epoch)
        writer.add_scalar(f"R2/{dataloadertype}", _r2, epoch)
    return output
