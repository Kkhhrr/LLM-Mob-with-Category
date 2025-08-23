import pandas as pd
import numpy as np
import geopandas as gpd
from tqdm import tqdm
from pathlib import Path
import pickle as pickle
from shapely import wkt

from sklearn.preprocessing import OrdinalEncoder
import os

from joblib import Parallel, delayed
from joblib import parallel_backend
import torch
from torch.nn.utils.rnn import pad_sequence

import trackintel as ti


class sp_loc_dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        source_root,
        user=None,
        dataset="geolife",
        city=None,
        data_type="train",
        previous_day=7,
        model_type="transformer",
        day_selection="default",
    ):
        self.root = source_root
        self.user = user
        self.data_type = data_type
        self.previous_day = previous_day
        self.model_type = model_type
        self.dataset = dataset
        self.city = city
        self.day_selection = day_selection

        if self.dataset == "fsq" and not self.city:
            raise ValueError("错误: 当数据集为 'fsq' 时, 必须提供 city 参数 (例如 'tky' 或 'nyc').")

        if user is None:
            self.is_individual_model = False
        else:
            self.is_individual_model = True

        city_suffix = f"_{self.city}" if self.dataset == "fsq" else ""

        if self.is_individual_model:
            self.data_dir = os.path.join(
                source_root, "temp", "individual", f"{self.dataset}{city_suffix}_{self.model_type}_{previous_day}"
            )
            if not os.path.exists(self.data_dir):
                os.makedirs(self.data_dir)
            save_path = os.path.join(self.data_dir, f"{user}_{data_type}.pk")
        else:
            self.data_dir = os.path.join(source_root, "temp")
            if day_selection == "default":
                save_path = os.path.join(
                    self.data_dir, f"{self.dataset}{city_suffix}_{self.model_type}_{previous_day}_{data_type}.pk"
                )
            else:
                save_path = os.path.join(
                    self.data_dir, f"{self.dataset}{city_suffix}_{''.join(str(x) for x in self.day_selection)}_{data_type}.pk"
                )

        if Path(save_path).is_file():
            self.data = pickle.load(open(save_path, "rb"))
        else:
            parent = Path(save_path).parent.absolute()
            if not os.path.exists(parent):
                os.makedirs(parent)
            self.data = self.generate_data()

        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        selected = self.data[idx]
        return_dict = {}
        x = torch.tensor(selected["X"])
        y = torch.tensor(selected["Y"])
        
        # --- USER ID 修正 ---
        # 返回整个 user_X 数组，让 collate_fn 来统一处理
        return_dict["user"] = torch.tensor(selected["user_X"])
        
        return_dict["time"] = torch.tensor(selected["start_min_X"] // 30)
        return_dict["diff"] = torch.tensor(selected["diff"])
        return_dict["duration"] = torch.tensor(selected["dur_X"] // 30, dtype=torch.long)
        return_dict["weekday"] = torch.tensor(selected["weekday_X"])
        if self.dataset == "gc":
            return_dict["poi"] = torch.tensor(np.array(selected["poi_X"]), dtype=torch.float32)
        return x, y, return_dict

    def generate_data(self):
        if self.dataset == "gc":
            self.poi_data = load_pk_file(os.path.join(self.root, "poiValues_lda_16_500.pk"))
            self.valid_ids = load_pk_file(os.path.join(self.root, f"valid_ids_{self.dataset}.pk"))
            ori_data = pd.read_csv(os.path.join(self.root, f"dataset_{self.dataset}.csv"))
        elif self.dataset == "fsq":
            data_folder = os.path.join(self.root, self.dataset, self.city)
            self.valid_ids = load_pk_file(os.path.join(data_folder, f"valid_ids_foursquare_{self.city}.pk"))
            ori_data = pd.read_csv(os.path.join(data_folder, f"dataSet_foursquare_{self.city}.csv"))
        elif self.dataset == "geolife":
            data_folder = os.path.join(self.root, self.dataset)
            self.valid_ids = load_pk_file(os.path.join(data_folder, f"valid_ids_{self.dataset}.pk"))
            ori_data = pd.read_csv(os.path.join(data_folder, f"dataSet_{self.dataset}.csv"))
        else:
            self.valid_ids = load_pk_file(os.path.join(self.root, f"valid_ids_{self.dataset}.pk"))
            ori_data = pd.read_csv(os.path.join(self.root, f"dataset_{self.dataset}.csv"))

        ori_data.sort_values(by=["user_id", "start_day", "start_min"], inplace=True)
        if 'duration' in ori_data.columns:
            ori_data.loc[ori_data["duration"] > 60 * 24 * 2 - 1, "duration"] = 60 * 24 * 2 - 1
        train_data, vali_data, test_data = self._splitDataset(ori_data)

        if self.is_individual_model:
            total_num_location = train_data.groupby("user_id")["location_id"].max() + 1
            user_dict = total_num_location.to_dict()
            save_path = os.path.join(self.data_dir, f"loc.pk")
            save_pk_file(save_path, user_dict)
        else:
            train_data, vali_data, test_data, enc = self._encode_loc(train_data, vali_data, test_data)
            if self.dataset == "gc":
                self.poi_data["index"] = enc.transform(self.poi_data["index"].reshape(-1, 1)) + 2
                self.poi_data["index"] = np.squeeze(self.poi_data["index"], axis=-1)
                idx = np.where(self.poi_data["index"] == 1)[0]
                self.poi_data["poiValues"][idx, :, :] = np.zeros_like(self.poi_data["poiValues"][idx, :, :])

        train_records = self._preProcessDatasets(train_data, "train")
        validation_records = self._preProcessDatasets(vali_data, "validation")
        test_records = self._preProcessDatasets(test_data, "test")

        if self.data_type == "test":
            return test_records
        if self.data_type == "validation":
            return validation_records
        if self.data_type == "train":
            return train_records

    def _encode_loc(self, train, validation, test):
        enc = OrdinalEncoder(dtype=np.int64, handle_unknown="use_encoded_value", unknown_value=-1).fit(
            train["location_id"].values.reshape(-1, 1)
        )
        train["location_id"] = enc.transform(train["location_id"].values.reshape(-1, 1)) + 2
        validation["location_id"] = enc.transform(validation["location_id"].values.reshape(-1, 1)) + 2
        test["location_id"] = enc.transform(test["location_id"].values.reshape(-1, 1)) + 2
        return train, validation, test, enc

    def _splitDataset(self, totalData):
        totalData = totalData.groupby("user_id", group_keys=False).apply(self.__getSplitDaysUser)
        train_data = totalData.loc[totalData["Dataset"] == "train"].copy()
        vali_data = totalData.loc[totalData["Dataset"] == "vali"].copy()
        test_data = totalData.loc[totalData["Dataset"] == "test"].copy()
        train_data.drop(columns={"Dataset"}, inplace=True)
        vali_data.drop(columns={"Dataset"}, inplace=True)
        test_data.drop(columns={"Dataset"}, inplace=True)
        return train_data, vali_data, test_data

    def __getSplitDaysUser(self, df):
        maxDay = df["start_day"].max()
        train_split = maxDay * 0.6
        vali_split = maxDay * 0.8
        df["Dataset"] = "test"
        df.loc[df["start_day"] < train_split, "Dataset"] = "train"
        df.loc[
            (df["start_day"] >= train_split) & (df["start_day"] < vali_split),
            "Dataset",
        ] = "vali"
        if self.is_individual_model:
            enc = OrdinalEncoder(
                dtype=np.int64,
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ).fit(df.loc[df["Dataset"] == "train", "location_id"].values.reshape(-1, 1))
            df["location_id"] = enc.transform(df["location_id"].values.reshape(-1, 1)) + 2
        return df

    def _preProcessDatasets(self, data, dataset_type):
        valid_records = self.__getValidSequence(data)
        city_suffix = f"_{self.city}" if self.dataset == "fsq" else ""
        if self.is_individual_model:
            for i, records in enumerate(valid_records):
                save_path = os.path.join(self.data_dir, f"{i+1}_{dataset_type}.pk")
                save_pk_file(save_path, records)
            return_data = valid_records[self.user - 1]
        else:
            valid_records = [item for sublist in valid_records for item in sublist]
            if self.day_selection == "default":
                save_path = os.path.join(
                    self.data_dir, f"{self.dataset}{city_suffix}_{self.model_type}_{self.previous_day}_{dataset_type}.pk"
                )
            else:
                save_path = os.path.join(
                    self.data_dir, f"{self.dataset}{city_suffix}_{''.join(str(x) for x in self.day_selection)}_{dataset_type}.pk"
                )
            save_pk_file(save_path, valid_records)
            return_data = valid_records
        return return_data

    def __getValidSequence(self, input_df):
        valid_user_ls = applyParallel(input_df.groupby("user_id"), self.___getValidSequenceUser, n_jobs=-1)
        return valid_user_ls

    def ___getValidSequenceUser(self, df):
        df.reset_index(drop=True, inplace=True)
        data_single_user = []
        min_days = df["start_day"].min()
        df["diff_day"] = df["start_day"] - min_days
        for index, row in df.iterrows():
            if row["diff_day"] < self.previous_day:
                continue
            hist = df.iloc[:index]
            hist = hist.loc[(hist["start_day"] >= (row["start_day"] - self.previous_day))]
            if not (row["id"] in self.valid_ids):
                continue
            if self.day_selection != "default":
                hist["diff"] = row["diff_day"] - hist["diff_day"]
                hist = hist.loc[hist["diff"].isin(self.day_selection)]
                if len(hist) < 2:
                    continue
            
            data_dict = {}
            data_dict["X"] = hist["location_id"].values
            data_dict["user_X"] = hist["user_id"].values
            data_dict["weekday_X"] = hist["weekday"].values
            data_dict["start_min_X"] = hist["start_min"].values
            if 'duration' in hist.columns:
                data_dict["dur_X"] = hist["duration"].values
            else:
                data_dict["dur_X"] = np.zeros(len(hist), dtype=int)
            data_dict["diff"] = (row["diff_day"] - hist["diff_day"]).astype(int).values
            if self.dataset == "gc":
                data_dict["poi_X"] = self._getPOIRepresentation(data_dict["X"])
            data_dict["Y"] = int(row["location_id"])
            data_single_user.append(data_dict)
        return data_single_user

    def _getPOIRepresentation(self, loc_ls):
        poi_rep = []
        for loc in loc_ls:
            idx = np.where(self.poi_data["index"] == loc)[0][0]
            matrix = self.poi_data["poiValues"][idx, :, :]
            poi_rep.append(matrix)
        return poi_rep

def save_pk_file(save_path, data):
    with open(save_path, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pk_file(save_path):
    return pickle.load(open(save_path, "rb"))

def applyParallel(dfGrouped, func, n_jobs, print_progress=True, **kwargs):
    with parallel_backend("threading", n_jobs=n_jobs):
        df_ls = Parallel()(delayed(func)(group, **kwargs) for _, group in tqdm(dfGrouped, disable=not print_progress))
    return df_ls

def collate_fn(batch):
    x_batch, y_batch = [], []
    x_dict_batch = {}
    if batch:
        for key in batch[0][-1]:
            x_dict_batch[key] = []

    for src_sample, tgt_sample, return_dict in batch:
        x_batch.append(src_sample)
        y_batch.append(tgt_sample)
        for key, value in return_dict.items():
            x_dict_batch[key].append(value)
    
    # --- COLLATE 修正: 统一处理为 batch_first=True ---
    x_batch_padded = pad_sequence(x_batch, batch_first=True, padding_value=0)
    y_batch_tensor = torch.tensor(y_batch, dtype=torch.int64)

    for key, value_list in x_dict_batch.items():
        # 将列表中的每个 tensor/array 转换为 tensor
        tensor_list = [torch.tensor(v, dtype=torch.int64) for v in value_list]
        x_dict_batch[key] = pad_sequence(tensor_list, batch_first=True, padding_value=0)

    return x_batch_padded, y_batch_tensor, x_dict_batch


def test_dataloader(train_loader):
    batch_size = train_loader.batch_size
    x_shape = 0
    x_dict_shape = 0
    for batch_idx, (x, y, x_dict) in tqdm(enumerate(train_loader)):
        x_shape += x.shape[0]
        x_dict_shape += x_dict["duration"].shape[0]
    print(x_shape / len(train_loader))
    print(x_dict_shape / len(train_loader))

if __name__ == "__main__":
    source_root = r"./data/"
    dataset_train = sp_loc_dataset(source_root, dataset="fsq", city="tky", data_type="train", previous_day=7)
    kwds_train = {"shuffle": False, "num_workers": 0, "batch_size": 2}
    train_loader = torch.utils.data.DataLoader(dataset_train, collate_fn=collate_fn, **kwds_train)
    test_dataloader(train_loader)