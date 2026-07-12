import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import re
import os

STRUCTURE_ID = "structure_id"

_NON_DESCRIPTOR_META = {"direction", "stress_axis", "elapsed_s", "converged", "max_iters_used"}


def get_npy_shape(filepath: str):
    dat = np.load(filepath)
    return dat.shape


class DescDimReducer:
    def __init__(
        self,
        df: pd.DataFrame,
        target_columns: list = None,
        path_column: str = "npy_path",
    ):
        self.database = df
        exclude = _NON_DESCRIPTOR_META | {path_column} | set(target_columns or ["cii"])
        self.descriptors_columns = [x for x in self.database.columns if x not in exclude]
        self.scaler = StandardScaler()
        self.scaler.fit(df[self.descriptors_columns])

    def transform(self, df):
        if len(df) == 0:
            return df
        data = df[self.descriptors_columns]
        df_other = df.drop(self.descriptors_columns, axis=1)
        scaled_data = self.scaler.transform(data)
        df_scaled = pd.DataFrame(scaled_data)
        df_scaled = pd.concat([df_other, df_scaled.set_index(df_other.index)], axis=1)
        return df_scaled


class MDDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transform=None,
        use_descriptors: bool = False,
        descriptor_transform: DescDimReducer = None,
        target_columns: list = None,
        path_column: str = "npy_path",
    ):
        self.database = df
        self.transform = transform
        self.use_descriptors = use_descriptors
        self.target_columns = target_columns if target_columns is not None else ["cii"]
        self.path_column = path_column
        self.descriptors_columns = None
        if use_descriptors:
            exclude = _NON_DESCRIPTOR_META | {path_column} | set(self.target_columns)
            self.descriptors_columns = [x for x in self.database.columns if x not in exclude]
            if descriptor_transform is not None:
                self.database = descriptor_transform.transform(self.database)
                self.descriptors_columns = [x for x in self.database.columns if x not in exclude]
            print(self.descriptors_columns)
            print(len(self.descriptors_columns))

    def __len__(self):
        return len(self.database)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.database.iloc[idx]
        target = np.array(row[self.target_columns], dtype=np.float32)
        path = row[self.path_column]
        structure = np.load(path).astype("float32")
        if self.transform:
            structure = self.transform(structure)
        else:
            structure = np.expand_dims(structure, axis=0)
        if self.use_descriptors:
            descriptors = np.array(self.database[self.descriptors_columns].iloc[idx]).astype("float32")
            return structure, target, path, descriptors
        else:
            return structure, target, path


def get_structure_id_from_filepath(filepath):
    # remove '_axis-X.npy' where X is any single char form the filename and returns a filename
    # data/structures/foo_axis-a.npy will give foo
    return re.sub(r'_axis-.\.npy$', '', os.path.basename(filepath))


def split_into_folds_nostratification(df: pd.DataFrame, cv_folds: int, which_folds: list[int]):
    if cv_folds < 1:
        raise ValueError(f"cv_folds must be at least 1. {cv_folds} provided")
    if cv_folds == 1:
        dftrainval, df_test = train_test_split(df, test_size=0.2)
        df_train, df_val = train_test_split(dftrainval, test_size=0.2)
        yield df_train, df_val, df_test, 0
    else:
        kf = KFold(n_splits=cv_folds)
        for fold_id, (trainval_idx, test_idx) in enumerate(kf.split(df)):
            if fold_id in which_folds:
                train_idx, val_idx = train_test_split(trainval_idx, test_size=(1 - 0.2) * 0.2)
                assert len(set(train_idx).intersection(val_idx)) == 0, "Overlap between train and val"
                assert len(set(train_idx).intersection(test_idx)) == 0, "Overlap between train and test"
                assert len(set(val_idx).intersection(test_idx)) == 0, "Overlap between val and test"
                df_train = df.iloc[train_idx, :]
                df_val = df.iloc[val_idx, :]
                df_test = df.iloc[test_idx, :]
                yield df_train, df_val, df_test, fold_id

def split_into_folds_stratification(df: pd.DataFrame, cv_folds: int, which_folds: list[int], path_column: str = "npy_path"):
    def _structures_by_id(_df: pd.DataFrame, idlist):
        return _df.query(f'{STRUCTURE_ID} in @idlist')
    if cv_folds < 1:
        raise ValueError(f'cv_folds must be at least 1. {cv_folds} provided')
    df[STRUCTURE_ID] = [get_structure_id_from_filepath(x) for x in df[path_column]]
    structure_ids = np.array(df[STRUCTURE_ID].unique().tolist())
    if cv_folds == 1:
        trainval_structure_ids, test_structure_ids = train_test_split(structure_ids, test_size=0.2)
        train_structure_ids, val_structure_ids = train_test_split(trainval_structure_ids, test_size=0.2)
        df_train = _structures_by_id(_df=df, idlist=train_structure_ids)
        df_val = _structures_by_id(_df=df, idlist=val_structure_ids)
        df_test = _structures_by_id(_df=df, idlist=test_structure_ids)
        yield df_train, df_val, df_test, 0
    else:
        kf = KFold(n_splits=cv_folds)
        for fold_id, (trainval_idx, test_idx) in enumerate(kf.split(structure_ids)):
            if fold_id in which_folds:
                # here trainval_idx & test_idx are indices of structure_ids array!
                train_idx, val_idx = train_test_split(trainval_idx, test_size=(1 - 0.2) * 0.2)
                train_structure_ids = structure_ids[train_idx]
                val_structure_ids = structure_ids[val_idx]
                test_structure_ids = structure_ids[test_idx]
                assert len(set(train_structure_ids).intersection(val_structure_ids)) == 0, "Overlap between train and val"
                assert len(set(train_structure_ids).intersection(test_structure_ids)) == 0, "Overlap between train and test"
                assert len(set(val_structure_ids).intersection(test_structure_ids)) == 0, "Overlap between val and test"
                df_train = _structures_by_id(_df=df, idlist=train_structure_ids).drop([STRUCTURE_ID], axis=1)
                df_val = _structures_by_id(_df=df, idlist=val_structure_ids).drop([STRUCTURE_ID], axis=1)
                df_test = _structures_by_id(_df=df, idlist=test_structure_ids).drop([STRUCTURE_ID], axis=1)

                yield df_train, df_val, df_test, fold_id


