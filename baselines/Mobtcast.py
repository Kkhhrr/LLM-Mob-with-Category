import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import sys
import os
import json
import math
import argparse
from sklearn.metrics import f1_score, recall_score
import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from joblib import Parallel, delayed, parallel_backend

# 尝试导入项目工具模块
try:
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_dir = os.path.abspath(os.path.join(current_script_dir, '..'))
    if project_root_dir not in sys.path:
        sys.path.insert(0, project_root_dir)
    from utils.utils import setup_seed
except ImportError as e:
    print(f"导入 utils.utils 模块错误: {e}")
    def setup_seed(seed):
        print(f"Warning: utils.utils not found. Using a basic setup_seed function.")
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

# --- 本地定义数据加载相关函数和类 ---
def save_pk_file(save_path, data):
    with open(save_path, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pk_file(save_path):
    return pickle.load(open(save_path, "rb"))

def applyParallel(dfGrouped, func, n_jobs, print_progress=True, **kwargs):
    with parallel_backend("threading", n_jobs=n_jobs):
        df_ls = Parallel()(delayed(func)(group, **kwargs) for _, group in tqdm(dfGrouped, disable=not print_progress))
    return df_ls

# --- Geolife 新增：Duration分箱函数 (采用更合理的设计)---
def discretize_duration(duration_minutes):
    """将分钟为单位的停留时长分到更细致、合理的离散箱中。"""
    if duration_minutes < 10:       # Bin 1: Quick Stop (< 10 min)
        return 1
    elif duration_minutes < 60:     # Bin 2: Short Activity (10 - 60 min)
        return 2
    elif duration_minutes < 240:    # Bin 3: Medium Activity (1 - 4 hours)
        return 3
    elif duration_minutes < 600:    # Bin 4: Long Activity / Workday (4 - 10 hours)
        return 4
    elif duration_minutes < 1200:   # Bin 5: Overnight (10 - 20 hours)
        return 5
    else:                           # Bin 6: Extended Stay (> 20 hours)
        return 6

class sp_loc_dataset(Dataset):
    def __init__(self, source_root, dataset="fsq_nyc", data_type="train",
                 previous_day=7, model_type="mobtcast_fsq_optimized",
                 day_selection="default",
                 args_ref=None,
                 source_filename_prefix="foursquare_nyc"):
        self.root = source_root
        self.data_type = data_type
        self.previous_day = previous_day
        self.model_type = model_type
        self.dataset_name = dataset
        self.source_filename_prefix = source_filename_prefix
        self.day_selection = day_selection
        self.args = args_ref
        self.poi_coordinates_map = {}
        self.poi_categories_map = {}
        self.location_encoder = None
        self.category_encoder = None
        self.coordinate_scaler = None

        self.data_dir = os.path.join(source_root, f"temp_{self.model_type}")

        if self.day_selection == "default":
            save_path_filename = f"{self.dataset_name}_{self.model_type}_{self.previous_day}_{data_type}.pk"
        else:
            day_selection_str = ''.join(str(x) for x in self.day_selection)
            save_path_filename = f"{self.dataset_name}_{self.model_type}_{day_selection_str}_{self.previous_day}_{data_type}.pk"

        save_path = os.path.join(self.data_dir, save_path_filename)

        if Path(save_path).is_file() and not (self.args and self.args.force_regenerate_data):
            print(f"加载预处理数据从: {save_path}")
            loaded_data = pickle.load(open(save_path, "rb"))
            self.data = loaded_data['data']
            self.poi_coordinates_map = loaded_data.get('poi_coordinates_map', {})
            self.poi_categories_map = loaded_data.get('poi_categories_map', {})
            self.location_encoder = loaded_data.get('location_encoder')
            self.category_encoder = loaded_data.get('category_encoder')
            self.coordinate_scaler = loaded_data.get('coordinate_scaler')
            if not self.poi_coordinates_map and self.args and self.args.m1_use_real_poi_coords:
                 self._load_poi_coordinates_and_categories()
            if not self.poi_categories_map and self.args and self.args.m1_use_poi_categories:
                 self._load_poi_coordinates_and_categories()
            if self.args and self.args.m1_use_real_poi_coords and self.args.m1_normalize_coords and self.coordinate_scaler is None and self.data_type == 'train':
                print("警告: 坐标标准化器未从缓存加载，将在训练数据上重新拟合。")
        else:
            parent = Path(save_path).parent.absolute()
            if not os.path.exists(parent): os.makedirs(parent)
            print(f"预处理数据 {save_path} 未找到或被强制重新生成。正在生成...")
            self.data = self.generate_data()

        self.len = len(self.data)
        print(f"数据集 {self.dataset_name} ({self.data_type}) 加载完成, 样本数: {self.len}")

    def _load_poi_coordinates_and_categories(self):
        try:
            data_df_path = os.path.join(self.root, f"dataSet_{self.source_filename_prefix}.csv")
            if not os.path.exists(data_df_path):
                print(f"错误: 数据文件 {data_df_path} 未找到!"); return

            use_cols = ['location_id']
            if self.args and self.args.m1_use_real_poi_coords:
                use_cols.extend(['latitude', 'longitude'])

            temp_df_check = pd.read_csv(data_df_path, nrows=1)
            has_category_column = False
            if self.args and self.args.m1_use_poi_categories and self.args.poi_category_column_name:
                if self.args.poi_category_column_name in temp_df_check.columns:
                    use_cols.append(self.args.poi_category_column_name)
                    has_category_column = True
                else:
                    print(f"警告: 指定的POI类别列 '{self.args.poi_category_column_name}' 在 {data_df_path} 中未找到。")
                    if self.args: self.args.m1_use_poi_categories = False

            data_df = pd.read_csv(data_df_path, usecols=lambda c: c in use_cols)
            data_df.drop_duplicates(subset=['location_id'], inplace=True)

            if self.args and self.args.m1_use_real_poi_coords:
                if 'latitude' in data_df.columns and 'longitude' in data_df.columns:
                    self.poi_coordinates_map = data_df.set_index('location_id')[['latitude', 'longitude']].apply(tuple, axis=1).to_dict()
                    print(f"POI坐标地图已加载, {len(self.poi_coordinates_map)} 个独立POI。")
                else:
                    print(f"警告: 坐标列 'latitude' 或 'longitude' 未在 {data_df_path} 中找到。已禁用坐标相关功能。")
                    self.args.m1_use_real_poi_coords = False
                    self.poi_coordinates_map = {}

            if self.args and self.args.m1_use_poi_categories and has_category_column:
                unique_categories = data_df[self.args.poi_category_column_name].dropna().unique()
                self.category_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                encoded_categories = self.category_encoder.fit_transform(unique_categories.reshape(-1, 1)) + 1
                self.poi_categories_map = dict(zip(unique_categories, encoded_categories.flatten()))
                if self.args: self.args.m1_poi_category_vocab_size = len(unique_categories) + 1
                print(f"POI类别地图已加载并编码, {len(self.poi_categories_map)} 个独立类别, 词汇表大小: {self.args.m1_poi_category_vocab_size}。")

        except Exception as e:
            print(f"加载POI坐标/类别时出错: {e}");
            self.poi_coordinates_map = {}; self.poi_categories_map = {}
            if self.args:
                self.args.m1_use_real_poi_coords = False
                self.args.m1_use_poi_categories = False

    def __len__(self): return self.len

    def __getitem__(self, idx):
        selected = self.data[idx]
        x_ids = torch.tensor(selected["X_encoded"])
        y_id = torch.tensor(selected["Y_encoded"])
        return_dict = {"user": torch.tensor(selected["user_X"]),
                       "time": torch.tensor(selected["start_min_X"] // 30),
                       "diff": torch.tensor(selected["diff"]),
                       "weekday": torch.tensor(selected["weekday_X"])}

        if self.args and self.args.use_duration_feature:
            if "X_duration" in selected and selected["X_duration"] is not None:
                duration_bins = [discretize_duration(d) for d in selected["X_duration"]]
                return_dict["duration"] = torch.tensor(duration_bins, dtype=torch.long)
            else:
                return_dict["duration"] = torch.full_like(x_ids, -1, dtype=torch.long)

        x_coords_list = []; y_coord_tuple = (0.0, 0.0)
        x_category_ids_list = [];
        if self.args:
            if self.args.m1_use_real_poi_coords:
                for original_loc_id in selected.get("X_original", []):
                    coords = list(self.poi_coordinates_map.get(original_loc_id, (0.0, 0.0)))
                    if self.args.m1_normalize_coords and self.coordinate_scaler:
                        coords = self.coordinate_scaler.transform(np.array(coords).reshape(1, -1))[0].tolist()
                    x_coords_list.append(coords)
                y_original_loc_id = selected.get("Y_original")
                if y_original_loc_id is not None:
                    y_coord_tuple = list(self.poi_coordinates_map.get(y_original_loc_id, (0.0, 0.0)))
                    if self.args.m1_normalize_coords and self.coordinate_scaler:
                         y_coord_tuple = self.coordinate_scaler.transform(np.array(y_coord_tuple).reshape(1, -1))[0].tolist()
                if not x_coords_list and len(selected["X_encoded"]) > 0:
                     x_coords_list = [[0.0, 0.0]] * len(selected["X_encoded"])

            if self.args.m1_use_poi_categories:
                x_original_cats_str_list = selected.get("X_original_categories_str", [])
                for cat_str in x_original_cats_str_list:
                    cat_id = self.poi_categories_map.get(str(cat_str), 0)
                    x_category_ids_list.append(cat_id)
                target_len_cat = len(selected.get("X_original",[]))
                if len(x_category_ids_list) != target_len_cat:
                    x_category_ids_list.extend([0] * (target_len_cat - len(x_category_ids_list)))
                    x_category_ids_list = x_category_ids_list[:target_len_cat]

                if not x_category_ids_list and len(selected["X_encoded"]) > 0:
                    x_category_ids_list = [0] * len(selected["X_encoded"])

        x_coords_tensor = torch.tensor(x_coords_list, dtype=torch.float32) if x_coords_list else torch.empty(len(x_ids) if len(x_ids)>0 else 0,2, dtype=torch.float32).fill_(0.0)
        y_coord_tensor = torch.tensor(y_coord_tuple, dtype=torch.float32)

        if not (self.args and self.args.m1_use_poi_categories):
            return_dict["semantic_input"] = return_dict["weekday"]
        else:
            if len(x_category_ids_list) != len(x_ids):
                x_category_ids_list.extend([0] * (len(x_ids) - len(x_category_ids_list)))
                x_category_ids_list = x_category_ids_list[:len(x_ids)]
            return_dict["semantic_input"] = torch.tensor(x_category_ids_list, dtype=torch.long)

        return x_ids, y_id, return_dict, x_coords_tensor, y_coord_tensor

    def generate_data(self):
        self._load_poi_coordinates_and_categories()

        if self.data_type == 'train' and self.args and self.args.m1_use_real_poi_coords and self.args.m1_normalize_coords:
            if self.poi_coordinates_map:
                all_raw_coords_for_scaling = np.array(list(self.poi_coordinates_map.values()))
                if len(all_raw_coords_for_scaling) > 0:
                    self.coordinate_scaler = StandardScaler()
                    self.coordinate_scaler.fit(all_raw_coords_for_scaling)
                    print("坐标标准化器已在整个数据集的POI坐标上拟合。")
                else:
                    print("警告: 未找到坐标数据来拟合标准化器 (在generate_data中)。")
            else:
                print("警告: POI坐标图谱为空，无法拟合坐标标准化器。")

        self.valid_ids = load_pk_file(os.path.join(self.root, f"valid_ids_{self.source_filename_prefix}.pk"))
        ori_data_path = os.path.join(self.root, f"dataSet_{self.source_filename_prefix}.csv")
        if not os.path.exists(ori_data_path): raise FileNotFoundError(f"原始数据文件未找到: {ori_data_path}")

        base_required_cols = ['user_id', 'start_day', 'start_min', 'location_id', 'weekday', 'id']
        read_cols = base_required_cols.copy()

        if self.args and self.args.m1_use_real_poi_coords:
            read_cols.extend(['latitude', 'longitude'])
        if self.args and self.args.m1_use_poi_categories and self.args.poi_category_column_name:
            temp_df_check = pd.read_csv(ori_data_path, nrows=1)
            if self.args.poi_category_column_name in temp_df_check.columns:
                 if self.args.poi_category_column_name not in read_cols:
                     read_cols.append(self.args.poi_category_column_name)

        if self.args and self.args.use_duration_feature:
            if 'duration' not in read_cols:
                read_cols.append('duration')

        ori_data = pd.read_csv(ori_data_path, usecols=lambda c: c in read_cols)

        final_required_cols = base_required_cols.copy()
        if self.args and self.args.m1_use_real_poi_coords:
            final_required_cols.extend(['latitude', 'longitude'])
        if self.args and self.args.use_duration_feature:
            final_required_cols.append('duration')

        for col in final_required_cols:
            if col not in ori_data.columns:
                raise ValueError(f"原始数据文件 {ori_data_path} 缺少必需列: {col}")

        ori_data.sort_values(by=["user_id", "start_day", "start_min"], inplace=True)
        train_data, vali_data, test_data = self._splitDataset(ori_data.copy())

        train_data["original_location_id"] = train_data["location_id"]
        vali_data["original_location_id"] = vali_data["location_id"]
        test_data["original_location_id"] = test_data["location_id"]

        if self.args and self.args.m1_use_poi_categories and self.args.poi_category_column_name in train_data.columns:
            train_data["original_category_str"] = train_data[self.args.poi_category_column_name].astype(str)
            vali_data["original_category_str"] = vali_data[self.args.poi_category_column_name].astype(str)
            test_data["original_category_str"] = test_data[self.args.poi_category_column_name].astype(str)

        train_data, vali_data, test_data, loc_encoder = self._encode_loc(train_data, vali_data, test_data)
        self.location_encoder = loc_encoder
        print(f"最大编码后位置 ID:{train_data.location_id_encoded.max()}, 唯一编码后位置 ID:{train_data.location_id_encoded.unique().shape[0]}")

        processed_data_for_current_type = self._preProcessDatasets(
            {"train": train_data, "validation": vali_data, "test": test_data}[self.data_type],
            self.data_type
        )

        data_to_save = {'data': processed_data_for_current_type,
                        'poi_coordinates_map': self.poi_coordinates_map,
                        'poi_categories_map': self.poi_categories_map,
                        'location_encoder': self.location_encoder,
                        'category_encoder': self.category_encoder,
                        'coordinate_scaler': self.coordinate_scaler}

        if self.day_selection == "default":
            save_path_filename_gen = f"{self.dataset_name}_{self.model_type}_{self.previous_day}_{self.data_type}.pk"
        else:
            day_selection_str_gen = ''.join(str(x) for x in self.day_selection)
            save_path_filename_gen = f"{self.dataset_name}_{self.model_type}_{day_selection_str_gen}_{self.previous_day}_{self.data_type}.pk"
        save_path_full = os.path.join(self.data_dir, save_path_filename_gen)

        save_pk_file(save_path_full, data_to_save)
        print(f"预处理数据已保存至: {save_path_full}")
        return processed_data_for_current_type

    def _encode_loc(self, train, validation, test):
        enc = OrdinalEncoder(dtype=np.int64, handle_unknown="use_encoded_value", unknown_value=-1).fit(train["location_id"].values.reshape(-1, 1))
        train["location_id_encoded"] = enc.transform(train["location_id"].values.reshape(-1, 1)) + 2
        validation["location_id_encoded"] = enc.transform(validation["location_id"].values.reshape(-1, 1)) + 2
        test["location_id_encoded"] = enc.transform(test["location_id"].values.reshape(-1, 1)) + 2
        return train, validation, test, enc

    def _splitDataset(self, totalData):
        totalData = totalData.groupby("user_id", group_keys=False).apply(self.__getSplitDaysUser)
        train_data = totalData.loc[totalData["Dataset"] == "train"].copy(); vali_data = totalData.loc[totalData["Dataset"] == "vali"].copy(); test_data = totalData.loc[totalData["Dataset"] == "test"].copy()
        train_data.drop(columns={"Dataset"}, inplace=True); vali_data.drop(columns={"Dataset"}, inplace=True); test_data.drop(columns={"Dataset"}, inplace=True)
        return train_data, vali_data, test_data

    def __getSplitDaysUser(self, df):
        maxDay = df["start_day"].max(); train_split = maxDay * 0.6; vali_split = maxDay * 0.8
        df["Dataset"] = "test"; df.loc[df["start_day"] < train_split, "Dataset"] = "train"; df.loc[(df["start_day"] >= train_split) & (df["start_day"] < vali_split), "Dataset"] = "vali"
        return df

    def _preProcessDatasets(self, data, dataset_type):
        valid_records = self.__getValidSequence(data)
        valid_records_flat = [item for sublist in valid_records if sublist for item in sublist]
        return valid_records_flat

    def __getValidSequence(self, input_df):
        is_foursquare = 'fsq' in self.args.dataset if self.args else False
        n_jobs_setting = 1 if is_foursquare else (1 if (os.cpu_count() is not None and os.cpu_count() < 4) else -1)
        valid_user_ls = applyParallel(input_df.groupby("user_id"), self.___getValidSequenceUser, n_jobs=n_jobs_setting, print_progress=False)
        return valid_user_ls

    def ___getValidSequenceUser(self, df_user):
        df_user.reset_index(drop=True, inplace=True); data_single_user = []
        min_days = df_user["start_day"].min(); df_user["diff_day"] = df_user["start_day"] - min_days
        for index, row in df_user.iterrows():
            if row["diff_day"] < self.previous_day: continue
            hist = df_user.iloc[:index]; hist = hist.loc[(hist["start_day"] >= (row["start_day"] - self.previous_day))]

            if hist.empty: continue
            if not (row["id"] in self.valid_ids): continue

            if self.day_selection != "default":
                hist["diff_select"] = row["diff_day"] - hist["diff_day"]
                hist = hist.loc[hist["diff_select"].isin(self.day_selection)]
                if len(hist) < 2: continue

            if len(hist) < 1: continue

            data_dict = {"X_encoded": hist["location_id_encoded"].values,
                         "X_original": hist["original_location_id"].values,
                         "user_X": hist["user_id"].values[0], "weekday_X": hist["weekday"].values,
                         "start_min_X": hist["start_min"].values,
                         "diff": (row["diff_day"] - hist["diff_day"]).astype(int).values,
                         "Y_encoded": int(row["location_id_encoded"]),
                         "Y_original": row["original_location_id"]}
            if self.args and self.args.m1_use_poi_categories and "original_category_str" in hist.columns:
                data_dict["X_original_categories_str"] = hist["original_category_str"].values
                data_dict["Y_original_category_str"] = row["original_category_str"]

            if self.args and self.args.use_duration_feature:
                data_dict["X_duration"] = hist["duration"].values

            data_single_user.append(data_dict)
        return data_single_user

# ============================================================================================
# START: GEOLIFE/OPTIMIZED MODEL AND HELPERS
# ============================================================================================

def local_collate_fn_geolife(batch, use_real_coords_flag):
    x_ids_batch, y_ids_batch = [], []; x_coords_batch_list, y_coords_batch_list = [], []
    x_dict_batch = {"len": []}
    if batch and len(batch[0]) >= 3 and isinstance(batch[0][2], dict):
        for key_in_dict in batch[0][2]: x_dict_batch[key_in_dict] = []
    for sample in batch:
        if len(sample) == 5: x_ids_item, y_id_item, x_features_dict_item, x_coords_item, y_coord_item = sample
        else: raise ValueError(f"local_collate_fn_geolife: __getitem__ 返回了意外数量的元素: {len(sample)}")
        x_ids_batch.append(x_ids_item); y_ids_batch.append(y_id_item)
        x_dict_batch["len"].append(len(x_ids_item))
        for key_in_dict_item in x_features_dict_item:
             if key_in_dict_item in x_dict_batch:
                x_dict_batch[key_in_dict_item].append(x_features_dict_item[key_in_dict_item])
        if use_real_coords_flag: x_coords_batch_list.append(x_coords_item); y_coords_batch_list.append(y_coord_item)
    x_ids_padded = pad_sequence(x_ids_batch, batch_first=False, padding_value=0)
    y_ids_tensor = torch.tensor(y_ids_batch, dtype=torch.int64)
    if "user" in x_dict_batch: x_dict_batch["user"] = torch.tensor(x_dict_batch["user"], dtype=torch.int64)
    x_dict_batch["len"] = torch.tensor(x_dict_batch["len"], dtype=torch.int64)
    for key_in_dict_final in x_dict_batch:
        if key_in_dict_final in ["user", "len", "history_count"]: continue
        if key_in_dict_final == "duration":
            x_dict_batch[key_in_dict_final] = pad_sequence(x_dict_batch[key_in_dict_final], batch_first=False, padding_value=0)
        elif key_in_dict_final == "semantic_input" and x_dict_batch[key_in_dict_final] and isinstance(x_dict_batch[key_in_dict_final][0], torch.Tensor):
            x_dict_batch[key_in_dict_final] = pad_sequence(x_dict_batch[key_in_dict_final], batch_first=False, padding_value=0)
        elif isinstance(x_dict_batch[key_in_dict_final], list) and len(x_dict_batch[key_in_dict_final]) > 0 and isinstance(x_dict_batch[key_in_dict_final][0], torch.Tensor):
             x_dict_batch[key_in_dict_final] = pad_sequence(x_dict_batch[key_in_dict_final], batch_first=False, padding_value=0)
    if use_real_coords_flag and x_coords_batch_list:
        valid_x_coords_list = [t for t in x_coords_batch_list if t.numel() > 0]
        if valid_x_coords_list:
            x_coords_padded = pad_sequence(valid_x_coords_list, batch_first=False, padding_value=0.0)
        else:
            dummy_seq_len = x_ids_padded.size(0); batch_s = len(y_ids_batch)
            x_coords_padded = torch.full((dummy_seq_len, batch_s, 2), 0.0, dtype=torch.float32)
        y_coords_tensor = torch.stack(y_coords_batch_list, dim=0) if y_coords_batch_list else torch.full((len(y_ids_batch), 2), 0.0, dtype=torch.float32)
    else:
        dummy_seq_len = x_ids_padded.size(0); batch_s = len(y_ids_batch)
        x_coords_padded = torch.full((dummy_seq_len, batch_s, 2), 0.0, dtype=torch.float32)
        y_coords_tensor = torch.full((batch_s, 2), 0.0, dtype=torch.float32)
    return x_ids_padded, y_ids_tensor, x_dict_batch, x_coords_padded, y_coords_tensor

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__(); self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model); position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term); pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1); self.register_buffer('pe', pe)
    def forward(self, x): x = x + self.pe[:x.size(0), :]; return self.dropout(x)

class TransformerSocial(nn.Module):
    def __init__(self, hidden_dim=256, nheads=2, dropout=0.2, num_encoder_layers=2, dim_feedforward=None):
        super(TransformerSocial, self).__init__()
        if dim_feedforward is None: dim_feedforward = 2 * hidden_dim
        encoder_layers = nn.TransformerEncoderLayer(hidden_dim, nheads, dim_feedforward, dropout, activation=F.relu, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.hidden_dim = hidden_dim
    def forward(self, x):
        transformer_out = self.transformer_encoder(x.permute(1,0,2))
        return transformer_out[-1, :, :].squeeze(0)

class TransformerBlock(nn.Module):
    def __init__(self, hour_len, hidden_dim=256, nheads=8, dropout=0.2, num_encoder_layers=6, dim_feedforward=None):
        super(TransformerBlock, self).__init__()
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout, max_len=hour_len)
        if dim_feedforward is None: dim_feedforward = 2 * hidden_dim
        # MODIFICATION: Changed activation to F.relu to match mobtcast2
        encoder_layers = nn.TransformerEncoderLayer(hidden_dim, nheads, dim_feedforward, dropout, activation=F.relu, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers); self.hidden_dim = hidden_dim
    def forward(self, x): h = x.permute(1, 0, 2); transformer_input = self.pos_encoder(h); transformer_out = self.transformer_encoder(transformer_input); return transformer_out[-1, :, :].squeeze(0)

class M1_geolife(nn.Module):
    def __init__(self, cfg_params, args_shim, model_args_main):
        super(M1_geolife, self).__init__(); self.__dict__.update(args_shim.__dict__); cfg_params.copyAttrib(self)
        self.use_duration_feature = model_args_main.use_duration_feature
        self.emb_poi = nn.Embedding(self.poi_num, self.embedding_dim, padding_idx=0)
        if self.use_real_poi_coords:
            coord_embed_dim = model_args_main.m1_coord_embedding_dim
            self.emb_loc = nn.Sequential(nn.Linear(2, coord_embed_dim), nn.LayerNorm(coord_embed_dim), nn.ReLU(), nn.Linear(coord_embed_dim, self.hidden_dim))
        else: self.emb_loc = nn.Linear(2, self.hidden_dim)
        self.emb_user = nn.Embedding(self.user_num, self.user_embed, padding_idx=0 if self.user_num > 0 else None)
        self.emb_semantic = nn.Embedding(self.category_num, model_args_main.m1_semantic_embed_dim, padding_idx=0)
        self.emb_tid = nn.Embedding(self.tid_num, model_args_main.m1_tid_embed_dim, padding_idx=0)
        if self.use_duration_feature:
            self.emb_duration = nn.Embedding(model_args_main.m1_duration_bin_num, model_args_main.m1_duration_embed_dim, padding_idx=0)
        
        # MODIFICATION: This logic is now adapted to handle cases without duration, just like mobtcast2's M1_Simple
        # We calculate the combined dimension of semantic and tid embeddings dynamically.
        if not self.use_duration_feature:
            semantic_tid_embed_combined_dim = self.hidden_dim - self.embedding_dim
            if semantic_tid_embed_combined_dim < 0:
                raise ValueError(f"hidden_dim ({self.hidden_dim}) must be >= embedding_dim ({self.embedding_dim})")
            # Re-initialize emb_semantic and emb_tid with calculated dimensions
            dim_part1 = semantic_tid_embed_combined_dim // 2
            dim_part2 = semantic_tid_embed_combined_dim - dim_part1
            self.emb_semantic = nn.Embedding(self.category_num, dim_part1, padding_idx=0)
            self.emb_tid = nn.Embedding(self.tid_num, dim_part2, padding_idx=0)
            self.input_feature_projection = None # Not needed in this case
        else: # Original logic for test.py if duration is used
            current_concat_dim = self.embedding_dim + model_args_main.m1_semantic_embed_dim + model_args_main.m1_tid_embed_dim
            if self.use_duration_feature:
                current_concat_dim += model_args_main.m1_duration_embed_dim
            self.input_feature_projection = nn.Linear(current_concat_dim, self.hidden_dim)

        tf_nheads = model_args_main.m1_transformer_nheads; tf_nlayers = model_args_main.m1_transformer_nlayers
        tf_dim_feedforward = model_args_main.m1_transformer_dim_feedforward
        self.transformer = TransformerBlock(hour_len=self.obs_len, hidden_dim=self.hidden_dim, nheads=tf_nheads, dropout=self.drop_out, num_encoder_layers=tf_nlayers, dim_feedforward=tf_dim_feedforward)
        self.transformer_loc = TransformerBlock(hour_len=self.obs_len, hidden_dim=self.hidden_dim, nheads=tf_nheads, dropout=self.drop_out, num_encoder_layers=tf_nlayers, dim_feedforward=tf_dim_feedforward)
        self.transformer_social = TransformerSocial(hidden_dim=self.hidden_dim, nheads=max(1,tf_nheads//2), dropout=self.drop_out, num_encoder_layers=max(1,tf_nlayers//2), dim_feedforward=tf_dim_feedforward)
        fc_input_dim = self.hidden_dim + self.user_embed
        if self.use_real_poi_coords : fc_input_dim += self.hidden_dim
        self.fc = nn.Linear(fc_input_dim, self.hidden_dim); self.fc_score = nn.Linear(self.hidden_dim, self.poi_num)
        self.fc_loc = nn.Linear(self.hidden_dim, 2); self.drop = nn.Dropout(self.drop_out)

    def forward(self, input_x_ids, input_user, input_semantic_val, input_tid_val, input_loc_coords_seq, input_social, social_tid, social_semantic, input_duration_val=None):
        user_emb = self.emb_user(input_user); poi_seq_emb = self.emb_poi(input_x_ids); semantic_seq_emb = self.emb_semantic(input_semantic_val); tid_seq_emb = self.emb_tid(input_tid_val)
        
        feature_list = [poi_seq_emb, semantic_seq_emb, tid_seq_emb]
        if self.use_duration_feature and input_duration_val is not None:
            duration_seq_emb = self.emb_duration(input_duration_val)
            feature_list.append(duration_seq_emb)
        
        tra_in_concat = torch.cat(feature_list, -1)

        if self.input_feature_projection: # Used if duration is enabled (original test.py logic)
             tra_in = self.input_feature_projection(tra_in_concat)
        else: # Used if duration is disabled (mobtcast2 logic)
            tra_in = tra_in_concat

        output_main_seq = self.transformer(tra_in)
        
        if self.use_real_poi_coords:
            if input_loc_coords_seq.numel() == 0 or input_loc_coords_seq.size(-1) != 2:
                dummy_coords = torch.zeros(input_x_ids.size(0), self.obs_len, 2, device=input_x_ids.device, dtype=torch.float32)
                loc_seq_emb_processed = self.emb_loc(dummy_coords)
            else:
                loc_seq_emb_processed = self.emb_loc(input_loc_coords_seq)
            output_loc_seq = self.transformer_loc(loc_seq_emb_processed)
            merge_features = torch.cat((output_main_seq, user_emb, output_loc_seq), -1)
        else:
            merge_features = torch.cat((output_main_seq, user_emb), -1)
        
        # MODIFICATION: Changed activation to F.relu to match mobtcast2
        pred_feature_fc_in = self.drop(F.relu(merge_features)); pred_feature_fc_out = self.fc(pred_feature_fc_in)
        # MODIFICATION: Changed activation to F.relu to match mobtcast2
        score_logits = self.fc_score(F.relu(pred_feature_fc_out)); log_softmax_score = F.log_softmax(score_logits, dim=1)
        # MODIFICATION: Changed activation to F.relu to match mobtcast2
        pred_next_loc_coords = self.fc_loc(F.relu(pred_feature_fc_out)); return log_softmax_score, pred_next_loc_coords

class MobTCastModelArgsShim_geolife:
    def __init__(self, user_args_main):
        self.embedding_dim = user_args_main.embed_dims['loc']; self.hidden_dim = user_args_main.hidden_size
        self.user_embed = user_args_main.embed_dims['user']; self.drop_out = user_args_main.dropout_rate
        self.use_cuda = (user_args_main.device == "cuda"); self.use_real_poi_coords = user_args_main.m1_use_real_poi_coords
class MobTCastCfgParamsShim_geolife:
    def __init__(self, user_args_main, loc_vocab_size_val, num_users_val):
        self.poi_num = loc_vocab_size_val; self.user_num = num_users_val
        if user_args_main.m1_use_poi_categories: self.category_num = user_args_main.m1_poi_category_vocab_size
        else: self.category_num = user_args_main.m1_weekday_category_num
        self.tid_num = user_args_main.m1_tid_num; self.obs_len = user_args_main.obs_len
        self.alpha = user_args_main.m1_alpha_loc_loss if user_args_main.m1_use_real_poi_coords else 0.0
        self.beta = user_args_main.m1_beta_loc_loss if user_args_main.m1_use_real_poi_coords else 0.0
    def copyAttrib(self, obj_to_copy_to):
        for key, value in self.__dict__.items(): setattr(obj_to_copy_to, key, value)

def _pad_or_truncate_tensor_geolife(tensor, target_len, pad_value=0, is_coords=False):
    current_len = tensor.size(0)
    if current_len == target_len: return tensor
    padding_dims = tensor.size()[1:]
    if current_len > target_len: return tensor[:target_len, ...]
    else:
        pad_size = target_len - current_len
        if tensor.dim() == 1 and not is_coords:
            padding = torch.full((pad_size,), int(pad_value), dtype=tensor.dtype, device=tensor.device)
        elif tensor.dim() == 2 and is_coords:
             padding = torch.full((pad_size, tensor.size(1)), float(pad_value), dtype=tensor.dtype, device=tensor.device)
        elif tensor.dim() >= 2 :
             padding_shape = (pad_size,) + tuple(padding_dims)
             if is_coords: padding = torch.full(padding_shape, float(pad_value), dtype=tensor.dtype, device=tensor.device)
             else: padding = torch.full(padding_shape, int(pad_value), dtype=tensor.dtype, device=tensor.device)
        else:
            raise ValueError(f"Unexpected tensor dim for padding: {tensor.dim()}")
        return torch.cat([tensor, padding], dim=0)

def custom_collate_fn_with_padding_geolife(batch_data_from_dataset, fixed_target_len, local_collate_fn_ref, use_poi_coords_flag_for_local_collate):
    x_loc_ids_b, y_loc_id_b, x_dict_b, x_loc_coords_b, y_loc_coords_b = \
        local_collate_fn_ref(batch_data_from_dataset, use_poi_coords_flag_for_local_collate)
    pad_val_id = 0
    pad_val_coord = 0.0
    x_loc_ids_b_final = _pad_or_truncate_tensor_geolife(x_loc_ids_b, fixed_target_len, pad_value=pad_val_id)
    for key_to_pad in ['time', 'semantic_input', 'diff', 'weekday', 'duration']:
        if key_to_pad in x_dict_b and x_dict_b[key_to_pad] is not None:
            if isinstance(x_dict_b[key_to_pad], torch.Tensor):
                 x_dict_b[key_to_pad] = _pad_or_truncate_tensor_geolife(x_dict_b[key_to_pad], fixed_target_len, pad_value=pad_val_id)
    x_loc_coords_b_final = _pad_or_truncate_tensor_geolife(x_loc_coords_b, fixed_target_len, pad_value=pad_val_coord, is_coords=True)
    return x_loc_ids_b_final, y_loc_id_b, x_dict_b, x_loc_coords_b_final, y_loc_coords_b

class MobTCast_Predictor_geolife:
    def __init__(self, loc_vocab_size, num_users, user_args_instance=None):
        if user_args_instance is None: raise ValueError("必须提供 user_args_instance")
        self.args = user_args_instance; self.device = self.args.device
        m1_args_shim = MobTCastModelArgsShim_geolife(self.args)
        m1_cfg_params_shim = MobTCastCfgParamsShim_geolife(self.args, loc_vocab_size, num_users)
        self.model = M1_geolife(cfg_params=m1_cfg_params_shim, args_shim=m1_args_shim, model_args_main=self.args).to(self.device)
        self.criterion_poi_id = nn.CrossEntropyLoss(ignore_index=0)
        self.criterion_loc_coords_mse = None; self.criterion_loc_coords_l1 = None
        if self.args.m1_use_real_poi_coords:
            if self.args.m1_alpha_loc_loss > 0: self.criterion_loc_coords_mse = nn.MSELoss(); print("已启用坐标预测辅助损失 (MSELoss, alpha)。")
            if self.args.m1_beta_loc_loss > 0: self.criterion_loc_coords_l1 = nn.L1Loss(); print("已启用坐标预测辅助损失 (L1Loss, beta)。")
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay, betas=(self.args.adam_beta1, self.args.adam_beta2), eps=self.args.adam_eps)
        self.lr_scheduler = self._get_lr_scheduler()
        self.start_epoch = 0; self.best_val_loss = float('inf')
        print(f"MobTCast (geolife) Predictor 初始化: loc_vocab_size={loc_vocab_size}, num_users={num_users}")
        if self.args.use_duration_feature: print(f"使用停留时长(duration)作为语义输入, 类别词汇表大小: {self.args.m1_duration_bin_num}")

    def _get_lr_scheduler(self):
        if self.args.lr_scheduler_type == 'reduce_on_plateau': return ReduceLROnPlateau(self.optimizer, mode='min', factor=self.args.lr_scheduler_factor, patience=self.args.lr_scheduler_patience)
        elif self.args.lr_scheduler_type == 'cosine_annealing_warmup':
            warmup_epochs = self.args.lr_warmup_epochs; total_epochs = self.args.epochs
            def lr_lambda(current_epoch):
                if current_epoch < warmup_epochs: return float(current_epoch + 1) / float(warmup_epochs +1)
                progress = float(current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs)); return 0.5 * (1.0 + math.cos(math.pi * progress))
            return LambdaLR(self.optimizer, lr_lambda)
        else: print(f"未知学习率调度器: {self.args.lr_scheduler_type}。"); return None

    def save_checkpoint(self, epoch, checkpoint_file_path, is_best=False):
        checkpoint = {'epoch': epoch, 'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'lr_scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler else None, 'best_val_loss': self.best_val_loss, 'user_args': self.args}
        if is_best: best_checkpoint_path = checkpoint_file_path.replace('.pth', '_best.pth'); torch.save(checkpoint, best_checkpoint_path); print(f"最佳模型检查点更新至 {best_checkpoint_path} (Epoch {epoch})")
        torch.save(checkpoint, checkpoint_file_path)

    def load_checkpoint(self, checkpoint_file_path, load_best=False):
        actual_path_to_load = checkpoint_file_path
        if load_best:
            best_path = checkpoint_file_path.replace('.pth', '_best.pth')
            actual_path_to_load = best_path if os.path.exists(best_path) else checkpoint_file_path
        if not os.path.exists(actual_path_to_load):
            print(f"检查点 {actual_path_to_load} 未找到。")
            return False
        try:
            print(f"从 {actual_path_to_load} 加载检查点...")
            checkpoint = torch.load(actual_path_to_load, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.lr_scheduler and 'lr_scheduler_state_dict' in checkpoint and checkpoint['lr_scheduler_state_dict'] is not None:
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            if not load_best:
                self.start_epoch = checkpoint.get('epoch', -1) + 1
                self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
                print(f"恢复训练: Epoch {self.start_epoch}, Best Val Loss: {self.best_val_loss:.4f}")
            else:
                print(f"最佳模型加载成功 (来自 Epoch {checkpoint.get('epoch', '未知')})。")
            return True
        except Exception as e:
            print(f"加载检查点 {actual_path_to_load} 失败: {e}。");
            return False

    def _prepare_batch_for_m1(self, x_loc_ids_b, x_dict_b, x_loc_coords_b):
        input_x_ids = x_loc_ids_b.transpose(0, 1).to(self.device).long()
        input_semantic_val = x_dict_b['semantic_input'].transpose(0, 1).to(self.device).long()
        input_tid_val = x_dict_b['time'].transpose(0, 1).to(self.device).long()
        input_user = x_dict_b['user'].to(self.device).long()
        input_poi_coords_seq = x_loc_coords_b.permute(1, 0, 2).to(self.device).float()
        input_duration_val = None
        if self.args.use_duration_feature and 'duration' in x_dict_b:
            input_duration_val = x_dict_b['duration'].transpose(0,1).to(self.device).long()
        current_obs_len = input_x_ids.size(1)
        assert current_obs_len == self.args.obs_len, f"序列长度不匹配: 期望 {self.args.obs_len}, 得到 {current_obs_len}"
        batch_size_current = input_x_ids.size(0)
        dummy_input_social = torch.zeros(batch_size_current, 1, current_obs_len, dtype=torch.long).to(self.device)
        dummy_social_tid = torch.zeros(batch_size_current, 1, current_obs_len, dtype=torch.long).to(self.device)
        dummy_social_semantic = torch.zeros(batch_size_current, 1, current_obs_len, dtype=torch.long).to(self.device)
        return input_x_ids, input_user, input_semantic_val, input_tid_val, input_poi_coords_seq, dummy_input_social, dummy_social_tid, dummy_social_semantic, input_duration_val

    def train(self, train_loader, eval_loader=None, checkpoint_file_path="checkpoint.pth"):
        epochs_no_improve_early_stopping = 0; checkpoint_dir = os.path.dirname(checkpoint_file_path)
        if checkpoint_dir and not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir, exist_ok=True)
        self.optimizer.zero_grad()
        for epoch in range(self.start_epoch, self.args.epochs):
            self.model.train(); total_loss_epoch = 0;
            pbar_desc = f"Epoch {epoch+1}/{self.args.epochs} [训练中]"; pbar = tqdm(train_loader, desc=pbar_desc, unit="batch")
            for batch_idx, (x_loc_ids_b, y_poi_id_b, x_dict_b, x_loc_coords_b, y_poi_coords_b) in enumerate(pbar):
                target_poi_ids = y_poi_id_b.to(self.device).long()
                target_poi_coords = y_poi_coords_b.to(self.device).float() # Get target coords
                prepared_inputs = self._prepare_batch_for_m1(x_loc_ids_b, x_dict_b, x_loc_coords_b)
                pred_log_softmax_poi_ids, pred_next_loc_coords = self.model(*prepared_inputs)
                
                loss = self.criterion_poi_id(pred_log_softmax_poi_ids, target_poi_ids)
                loss_poi_id_val = loss.item()
                loss_loc_coords_mse_val = 0.0
                loss_loc_coords_l1_val = 0.0
                
                # Add auxiliary loss if enabled
                if self.args.m1_use_real_poi_coords:
                    if self.criterion_loc_coords_mse:
                        loss_mse = self.criterion_loc_coords_mse(pred_next_loc_coords, target_poi_coords)
                        loss += self.args.m1_alpha_loc_loss * loss_mse
                        loss_loc_coords_mse_val = loss_mse.item()
                    if self.criterion_loc_coords_l1:
                        loss_l1 = self.criterion_loc_coords_l1(pred_next_loc_coords, target_poi_coords)
                        loss += self.args.m1_beta_loc_loss * loss_l1
                        loss_loc_coords_l1_val = loss_l1.item()
                
                loss.backward()
                self.optimizer.step(); self.optimizer.zero_grad()
                total_loss_epoch += loss.item()

                postfix_str = f"L_total={loss.item():.4f}|L_poi={loss_poi_id_val:.4f}"
                if self.args.m1_use_real_poi_coords:
                    if self.criterion_loc_coords_mse: postfix_str += f"|L_mse={loss_loc_coords_mse_val:.4f}"
                    if self.criterion_loc_coords_l1: postfix_str += f"|L_l1={loss_loc_coords_l1_val:.4f}"
                pbar.set_postfix_str(postfix_str)
            
            avg_train_loss = total_loss_epoch / len(train_loader) if len(train_loader) > 0 else 0
            print(f"Epoch {epoch+1}/{self.args.epochs}, 平均训练损失: {avg_train_loss:.4f}")

            if self.lr_scheduler and self.args.lr_scheduler_type == 'cosine_annealing_warmup': self.lr_scheduler.step()

            if eval_loader:
                self.model.eval(); total_eval_loss = 0
                with torch.no_grad():
                    for x_loc_ids_b, y_poi_id_b, x_dict_b, x_loc_coords_b, y_poi_coords_b in eval_loader:
                        target_poi_ids = y_poi_id_b.to(self.device).long()
                        target_poi_coords = y_poi_coords_b.to(self.device).float()
                        prepared_inputs = self._prepare_batch_for_m1(x_loc_ids_b, x_dict_b, x_loc_coords_b)
                        pred_log_softmax_poi_ids, pred_next_loc_coords = self.model(*prepared_inputs)
                        loss = self.criterion_poi_id(pred_log_softmax_poi_ids, target_poi_ids)

                        # Also calculate aux loss for validation
                        if self.args.m1_use_real_poi_coords:
                            if self.criterion_loc_coords_mse:
                                loss += self.args.m1_alpha_loc_loss * self.criterion_loc_coords_mse(pred_next_loc_coords, target_poi_coords)
                            if self.criterion_loc_coords_l1:
                                loss += self.args.m1_beta_loc_loss * self.criterion_loc_coords_l1(pred_next_loc_coords, target_poi_coords)
                        
                        total_eval_loss += loss.item()

                avg_eval_loss = total_eval_loss / len(eval_loader) if len(eval_loader) > 0 else float('inf')
                print(f"Epoch {epoch+1}/{self.args.epochs}, 验证损失: {avg_eval_loss:.4f}")

                if self.lr_scheduler and self.args.lr_scheduler_type == 'reduce_on_plateau': self.lr_scheduler.step(avg_eval_loss)

                is_best = avg_eval_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = avg_eval_loss
                    epochs_no_improve_early_stopping = 0
                    print(f"验证损失提升至 {self.best_val_loss:.4f}。")
                    self.save_checkpoint(epoch, checkpoint_file_path, is_best=True)
                else:
                    epochs_no_improve_early_stopping += 1
                    print(f"验证损失在 {epochs_no_improve_early_stopping} 个 epoch 内未提升。")

                if epoch % 5 == 0 or epoch == self.args.epochs - 1 or is_best: self.save_checkpoint(epoch, checkpoint_file_path, is_best=False)

                if epochs_no_improve_early_stopping >= self.args.patience:
                    print(f"早停在 epoch {epoch+1}。最佳验证损失: {self.best_val_loss:.4f}")
                    break
        print("训练完成。")

    def predict(self, test_loader, topk=1):
        self.model.eval(); all_preds_poi_ids = []; all_target_poi_ids = []
        with torch.no_grad():
            for x_loc_ids_b, y_poi_id_b, x_dict_b, x_loc_coords_b, y_poi_coords_b in tqdm(test_loader, desc="预测中", unit="batch"):
                target_poi_ids_cpu = y_poi_id_b.cpu().tolist()
                prepared_inputs = self._prepare_batch_for_m1(x_loc_ids_b, x_dict_b, x_loc_coords_b)
                pred_log_softmax_poi_ids, _ = self.model(*prepared_inputs)
                _, top_indices_poi_ids = torch.topk(pred_log_softmax_poi_ids, topk, dim=1)
                all_preds_poi_ids.extend(top_indices_poi_ids.cpu().tolist()); all_target_poi_ids.extend(target_poi_ids_cpu)
        return all_preds_poi_ids, all_target_poi_ids

# ============================================================================================
# END: GEOLIFE/OPTIMIZED MODEL
# ============================================================================================


# ============================================================================================
# START: FSQ MODEL AND HELPERS
# ============================================================================================

class PositionalEncoding_fsq(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding_fsq, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerBlock_fsq(nn.Module):
    def __init__(self, hour_len, hidden_dim=256, nheads=8, dropout=0.2, num_encoder_layers=6):
        super(TransformerBlock_fsq, self).__init__()
        self.pos_encoder = PositionalEncoding_fsq(hidden_dim, dropout, max_len=hour_len)
        encoder_layers = nn.TransformerEncoderLayer(hidden_dim, nheads, 2 * hidden_dim, dropout, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.hidden_dim = hidden_dim
    def forward(self, x):
        h = x.permute(1, 0, 2)
        transformer_input = self.pos_encoder(h)
        transformer_out = self.transformer_encoder(transformer_input)
        return transformer_out[-1, :, :].squeeze(0)

class TransformerSocial_fsq(nn.Module):
    def __init__(self, hidden_dim=256, nheads=2, dropout=0.2, num_encoder_layers=2):
        super(TransformerSocial_fsq, self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(hidden_dim, nheads, 2 * hidden_dim, dropout, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
    def forward(self, x):
        h = x.permute(1, 0, 2)
        transformer_out = self.transformer_encoder(h)
        return transformer_out[-1, :, :].squeeze(0)

class M1_fsq(nn.Module):
    def __init__(self, cfg_params, args):
        super(M1_fsq, self).__init__()
        self.__dict__.update(args.__dict__)
        cfg_params.copyAttrib(self)
        self.emb_poi = nn.Embedding(self.poi_num, self.embedding_dim)
        self.emb_loc = nn.Linear(2, self.hidden_dim)
        self.emb_user = nn.Embedding(self.user_num, self.user_embed)
        semantic_tid_embed_combined_dim = self.hidden_dim - self.embedding_dim
        if semantic_tid_embed_combined_dim < 0:
            raise ValueError(f"hidden_dim ({self.hidden_dim}) must be >= embedding_dim ({self.embedding_dim}) for semantic/tid embeddings.")
        dim_part1 = semantic_tid_embed_combined_dim // 2
        dim_part2 = semantic_tid_embed_combined_dim - dim_part1
        self.emb_semantic = nn.Embedding(self.category_num, dim_part1)
        self.emb_tid = nn.Embedding(self.tid_num, dim_part2)
        self.transformer = TransformerBlock_fsq(hour_len=self.obs_len, hidden_dim=self.hidden_dim)
        self.transformer_loc = TransformerBlock_fsq(hour_len=self.obs_len, hidden_dim=self.hidden_dim)
        self.transformer_social = TransformerSocial_fsq(hidden_dim=self.hidden_dim)
        fc_input_dim = self.hidden_dim + self.user_embed + self.hidden_dim
        self.fc = nn.Linear(fc_input_dim, self.hidden_dim)
        self.fc_score = nn.Linear(self.hidden_dim, self.poi_num)
        self.fc_loc = nn.Linear(self.hidden_dim, 2)
        self.drop = nn.Dropout(self.drop_out)

    def forward(self, input_x, input_user, input_semantic, input_tid, input_loc, input_social, social_tid, social_semantic):
        user_emb = self.emb_user(input_user)
        poi_seq_emb = self.emb_poi(input_x)
        semantic_seq_emb = self.emb_semantic(input_semantic)
        tid_seq_emb = self.emb_tid(input_tid)
        tra_in = torch.cat((poi_seq_emb, semantic_seq_emb, tid_seq_emb), -1)
        if tra_in.size(-1) != self.hidden_dim:
            raise ValueError(f"Concatenated input dimension ({tra_in.size(-1)}) does not match model hidden_dim ({self.hidden_dim}).")
        loc_emb = self.emb_loc(input_loc)
        output_main_seq = self.transformer(tra_in)
        output_loc_seq = self.transformer_loc(loc_emb)
        merge_features = torch.cat((output_main_seq, user_emb, output_loc_seq), -1)
        pred_feature_fc_in = self.drop(F.relu(merge_features))
        pred_feature_fc_out = self.fc(pred_feature_fc_in)
        score = self.fc_score(F.relu(pred_feature_fc_out))
        log_softmax_score = F.log_softmax(score, dim=1)
        pred_loc_aux = self.fc_loc(F.relu(pred_feature_fc_out))
        return log_softmax_score, pred_loc_aux

class MobTCastModelArgsShim_fsq:
    def __init__(self, user_args):
        self.embedding_dim = user_args.embed_dims['loc']
        self.hidden_dim = user_args.hidden_size
        self.user_embed = user_args.embed_dims['user']
        self.drop_out = user_args.dropout_rate
        self.use_cuda = (user_args.device == "cuda")

class MobTCastCfgParamsShim_fsq:
    def __init__(self, user_args, loc_vocab_size_val, num_users_val):
        self.poi_num = loc_vocab_size_val
        self.user_num = num_users_val
        self.category_num = user_args.m1_weekday_category_num
        self.tid_num = user_args.m1_tid_num
        self.obs_len = user_args.obs_len
        self.alpha = 0.0
        self.beta = 0.0
    def copyAttrib(self, obj_to_copy_to):
        for key, value in self.__dict__.items(): setattr(obj_to_copy_to, key, value)

def _pad_or_truncate_tensor_fsq(tensor, target_len, pad_value=0):
    current_len = tensor.size(0)
    if current_len == target_len:
        return tensor
    elif current_len > target_len:
        return tensor[:target_len, ...]
    else:
        pad_size = target_len - current_len
        padding_dims = tensor.size()[1:]
        padding = torch.full((pad_size, *padding_dims), pad_value, dtype=tensor.dtype, device=tensor.device)
        return torch.cat([tensor, padding], dim=0)

def local_collate_fn_fsq(batch):
    x_ids_batch, y_ids_batch, x_dict_batch = [], [], {"time": [], "weekday": [], "user": []}
    for sample in batch:
        x_ids_item, y_id_item, x_features_dict_item, _, _ = sample
        x_ids_batch.append(x_ids_item)
        y_ids_batch.append(y_id_item)
        x_dict_batch["time"].append(x_features_dict_item["time"])
        x_dict_batch["weekday"].append(x_features_dict_item["weekday"])
        # *** KEY FIX: Changed .[0] to .item() to handle 0-dim tensor ***
        x_dict_batch["user"].append(x_features_dict_item["user"].item())
    
    x_ids_padded = pad_sequence(x_ids_batch, batch_first=False, padding_value=0)
    time_padded = pad_sequence(x_dict_batch["time"], batch_first=False, padding_value=0)
    weekday_padded = pad_sequence(x_dict_batch["weekday"], batch_first=False, padding_value=0)
    y_ids_tensor = torch.tensor(y_ids_batch, dtype=torch.int64)
    user_tensor = torch.tensor(x_dict_batch["user"], dtype=torch.int64)
    final_dict = {"time": time_padded, "weekday": weekday_padded, "user": user_tensor}
    return x_ids_padded, y_ids_tensor, final_dict

def custom_collate_fn_with_padding_fsq(batch_data_from_dataset, fixed_target_len, original_collate_function_ref):
    x_loc_b, y_loc_b, x_dict_b = original_collate_function_ref(batch_data_from_dataset)
    pad_val = 0
    x_loc_b = _pad_or_truncate_tensor_fsq(x_loc_b, fixed_target_len, pad_value=pad_val)
    if 'time' in x_dict_b and x_dict_b['time'] is not None:
        x_dict_b['time'] = _pad_or_truncate_tensor_fsq(x_dict_b['time'], fixed_target_len, pad_value=pad_val)
    if 'weekday' in x_dict_b and x_dict_b['weekday'] is not None:
        x_dict_b['weekday'] = _pad_or_truncate_tensor_fsq(x_dict_b['weekday'], fixed_target_len, pad_value=pad_val)
    return x_loc_b, y_loc_b, x_dict_b

class MobTCast_Predictor_fsq:
    def __init__(self, loc_vocab_size, num_users, user_args_instance=None):
        if user_args_instance is None: raise ValueError("必须提供 user_args_instance")
        self.args = user_args_instance; self.device = self.args.device
        m1_args_shim = MobTCastModelArgsShim_fsq(self.args)
        m1_cfg_params_shim = MobTCastCfgParamsShim_fsq(self.args, loc_vocab_size, num_users)
        self.model = M1_fsq(cfg_params=m1_cfg_params_shim, args=m1_args_shim).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=self.args.lr_scheduler_factor, patience=self.args.lr_scheduler_patience)
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        print(f"MobTCast (FSQ) Predictor 初始化: loc_vocab_size={loc_vocab_size}, num_users={num_users}")

    def save_checkpoint(self, epoch, checkpoint_file_path, is_best=False):
        checkpoint = {'epoch': epoch, 'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'scheduler_state_dict': self.scheduler.state_dict(), 'best_val_loss': self.best_val_loss, 'user_args': self.args}
        if is_best:
            best_checkpoint_path = checkpoint_file_path.replace('.pth', '_best.pth')
            torch.save(checkpoint, best_checkpoint_path)
            print(f"最佳模型检查点已更新至 {best_checkpoint_path} (Epoch {epoch})")
        torch.save(checkpoint, checkpoint_file_path)

    def load_checkpoint(self, checkpoint_file_path, load_best=False):
        actual_path_to_load = checkpoint_file_path
        if load_best:
            best_path = checkpoint_file_path.replace('.pth', '_best.pth')
            if os.path.exists(best_path): actual_path_to_load = best_path
        if not os.path.exists(actual_path_to_load):
            print(f"检查点文件 {actual_path_to_load} 未找到。"); return False
        try:
            print(f"正在从 {actual_path_to_load} 加载检查点...");
            checkpoint = torch.load(actual_path_to_load, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint and self.scheduler is not None: self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if not load_best: self.start_epoch = checkpoint.get('epoch', -1) + 1; self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            return True
        except Exception as e:
            print(f"加载检查点 {actual_path_to_load} 失败: {e}。"); return False

    def _prepare_batch_for_m1(self, x_loc_batch_padded, x_dict_batch_padded):
        input_x = x_loc_batch_padded.transpose(0, 1).to(self.device).long()
        input_semantic = x_dict_batch_padded['weekday'].transpose(0, 1).to(self.device).long()
        input_tid = x_dict_batch_padded['time'].transpose(0, 1).to(self.device).long()
        input_user = x_dict_batch_padded['user'].to(self.device).long()
        batch_size_current = input_x.size(0)
        current_obs_len = input_x.size(1)
        assert current_obs_len == self.args.obs_len
        dummy_input_loc = torch.zeros(batch_size_current, current_obs_len, 2, dtype=torch.float32).to(self.device)
        dummy_input_social = torch.zeros(batch_size_current, 1, current_obs_len, dtype=torch.long).to(self.device)
        dummy_social_tid = torch.zeros(batch_size_current, 1, current_obs_len, dtype=torch.long).to(self.device)
        dummy_social_semantic = torch.zeros(batch_size_current, 1, current_obs_len, dtype=torch.long).to(self.device)
        return input_x, input_user, input_semantic, input_tid, dummy_input_loc, dummy_input_social, dummy_social_tid, dummy_social_semantic

    def train(self, train_loader, eval_loader=None, checkpoint_file_path="checkpoint.pth"):
        epochs_no_improve_early_stopping = 0
        checkpoint_dir = os.path.dirname(checkpoint_file_path)
        if checkpoint_dir and not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir, exist_ok=True)
        for epoch in range(self.start_epoch, self.args.epochs):
            self.model.train(); total_loss = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs} [训练中]", unit="batch")
            for x_loc_batch, y_batch, x_dict_batch in pbar:
                target_y = y_batch.to(self.device).long()
                prepared_inputs = self._prepare_batch_for_m1(x_loc_batch, x_dict_batch)
                self.optimizer.zero_grad()
                score, _ = self.model(*prepared_inputs)
                loss = self.criterion(score, target_y)
                loss.backward()
                if self.args.gradient_clip_value > 0: torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.gradient_clip_value)
                self.optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{self.optimizer.param_groups[0]['lr']:.1e}")
            avg_train_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
            print(f"Epoch {epoch+1}/{self.args.epochs}, 平均训练损失: {avg_train_loss:.4f}")

            if eval_loader:
                self.model.eval(); total_eval_loss = 0
                with torch.no_grad():
                    for x_loc_batch, y_batch, x_dict_batch in eval_loader:
                        target_y = y_batch.to(self.device).long()
                        prepared_inputs = self._prepare_batch_for_m1(x_loc_batch, x_dict_batch)
                        score, _ = self.model(*prepared_inputs)
                        loss = self.criterion(score, target_y)
                        total_eval_loss += loss.item()
                avg_eval_loss = total_eval_loss / len(eval_loader) if len(eval_loader) > 0 else float('inf')
                print(f"Epoch {epoch+1}/{self.args.epochs}, 验证损失: {avg_eval_loss:.4f}")
                self.scheduler.step(avg_eval_loss)

                is_best = avg_eval_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = avg_eval_loss
                    epochs_no_improve_early_stopping = 0
                    print(f"验证损失提升至 {self.best_val_loss:.4f}。")
                    self.save_checkpoint(epoch, checkpoint_file_path, is_best=True)
                else:
                    epochs_no_improve_early_stopping += 1
                    print(f"验证损失在 {epochs_no_improve_early_stopping} 个 epoch 内未提升。")

                if epoch % 5 == 0 or epoch == self.args.epochs - 1 or is_best: self.save_checkpoint(epoch, checkpoint_file_path, is_best=False)

                if epochs_no_improve_early_stopping >= self.args.patience:
                    print(f"早停在 epoch {epoch+1}。最佳验证损失: {self.best_val_loss:.4f}")
                    break
        print("训练完成。")

    def predict(self, test_loader, topk=1):
        self.model.eval(); all_preds = []; all_targets = []
        with torch.no_grad():
            for x_loc_batch, y_batch, x_dict_batch in tqdm(test_loader, desc="预测中", unit="batch"):
                targets_cpu = y_batch.cpu().tolist()
                prepared_inputs = self._prepare_batch_for_m1(x_loc_batch, x_dict_batch)
                score, _ = self.model(*prepared_inputs)
                _, top_indices = torch.topk(score, topk, dim=1)
                all_preds.extend(top_indices.cpu().tolist())
                all_targets.extend(targets_cpu)
        return all_preds, all_targets

# ============================================================================================
# END: FSQ MODEL
# ============================================================================================

# --- SHARED UTILITY FUNCTIONS ---
def calculate_all_metrics(all_preds, all_targets, acc_topk_list, ndcg_k=10):
    metrics_results = {}; num_samples = len(all_targets)
    if num_samples == 0:
        for k_val in acc_topk_list: metrics_results[f'Acc@{k_val}'] = 0.0
        metrics_results.update({'F1': 0.0, 'Recall': 0.0, f'NDCG@{ndcg_k}': 0.0, 'MRR': 0.0}); return metrics_results
    correct_counts = {k: 0 for k in acc_topk_list}; ndcg_scores, rr_scores = [], []
    top1_preds = [p[0] if (p and len(p)>0) else -1 for p in all_preds]
    for i in range(num_samples):
        actual = all_targets[i]; predicted_k = all_preds[i]
        for k_val in acc_topk_list:
            if actual in predicted_k[:k_val]: correct_counts[k_val] += 1
        preds_for_ndcg_mrr = predicted_k[:min(len(predicted_k), ndcg_k if ndcg_k > 0 else len(predicted_k))]
        try: rank = preds_for_ndcg_mrr.index(actual) + 1; ndcg_scores.append(1.0 / np.log2(rank + 1)); rr_scores.append(1.0 / rank)
        except ValueError: ndcg_scores.append(0.0); rr_scores.append(0.0)
    for k_val in acc_topk_list: metrics_results[f'Acc@{k_val}'] = (correct_counts[k_val]/num_samples) if num_samples else 0.0
    valid_targets = []; valid_top1_preds = []
    for t, p1 in zip(all_targets, top1_preds):
        if p1 != -1 : valid_targets.append(t); valid_top1_preds.append(p1)
    if not valid_targets: f1_w = 0.0; recall_w = 0.0
    else: f1_w = f1_score(valid_targets, valid_top1_preds, average="weighted", zero_division=0); recall_w = recall_score(valid_targets, valid_top1_preds, average="weighted", zero_division=0)
    metrics_results.update({'F1': f1_w, 'Recall': recall_w, f'NDCG@{ndcg_k}': np.mean(ndcg_scores) if ndcg_scores else 0.0, 'MRR': np.mean(rr_scores) if rr_scores else 0.0})
    return metrics_results

def get_max_user_id_and_loc_vocab_size(dataset_instance, data_root, source_filename_prefix, default_num_users_config):
    print("正在确定位置词汇表大小和最大用户ID...")
    max_loc_id = 0; default_num_users = default_num_users_config
    try:
        ori_data_path = os.path.join(data_root, f"dataSet_{source_filename_prefix}.csv")
        if os.path.exists(ori_data_path):
            ori_df = pd.read_csv(ori_data_path, usecols=['user_id'])
            if 'user_id' in ori_df.columns: max_user_id_from_file = ori_df['user_id'].max(); num_users = int(max_user_id_from_file) + 1; print(f"从原始文件 '{ori_data_path}' 推断的最大用户ID: {max_user_id_from_file}, 用户数: {num_users}")
            else: print(f"警告: {ori_data_path} 中未找到 'user_id' 列。"); num_users = default_num_users
        else: print(f"警告: {ori_data_path} 未找到。"); num_users = default_num_users
    except Exception as e: print(f"读取用户数出错: {e}。"); num_users = default_num_users
    if not hasattr(dataset_instance, 'data') or not dataset_instance.data:
        try: _ = dataset_instance[0]
        except Exception as e: print(f"访问数据集出错: {e}"); return 0, num_users, 0
    if not hasattr(dataset_instance, 'data') or not dataset_instance.data: print("错误: 数据集仍然为空，无法确定词汇表大小。"); return 0, num_users, 0
    max_user_id_in_data = 0; max_time_diff_observed = 0
    print("正在从已加载的预处理数据中扫描最大编码后位置ID...")
    for record in tqdm(dataset_instance.data, desc="扫描数据记录"):
        if 'X_encoded' in record and record['X_encoded'] is not None and len(record['X_encoded']) > 0: current_max_x_id = np.max(record['X_encoded']); max_loc_id = max(max_loc_id, current_max_x_id)
        if 'Y_encoded' in record and record['Y_encoded'] is not None: current_y_id = record['Y_encoded']; max_loc_id = max(max_loc_id, current_y_id)
        if 'user_X' in record and record['user_X'] is not None: max_user_id_in_data = max(max_user_id_in_data, record['user_X'])
        if 'diff' in record and record['diff'] is not None and len(record['diff']) > 0: max_time_diff_observed = max(max_time_diff_observed, np.max(record['diff']))
    if num_users <= max_user_id_in_data : print(f"警告：数据中最大用户ID ({max_user_id_in_data}) >= 推断用户数 ({num_users-1})。将使用 {max_user_id_in_data + 1}。"); num_users = int(max_user_id_in_data) + 1
    loc_vocab_size = int(max_loc_id) + 1
    print(f"从数据推断的最大编码后位置ID: {max_loc_id}。位置词汇表大小 (用于POI嵌入): {loc_vocab_size}")
    print(f"最终用户嵌入层用户数量: {num_users}");
    return loc_vocab_size, num_users, int(max_time_diff_observed)

# --- 主执行模块 ---
if __name__ == '__main__':
    DATASET_CONFIGS = {
        "fsq": {"default_num_users": 600, "user_embed_dim": 64, "loc_embed_dim": 128},
        "geolife": {"default_num_users": 200, "user_embed_dim": 32, "loc_embed_dim": 64,
                    "semantic_embed_dim": 32, "tid_embed_dim": 16,
                    "coord_embed_dim_mlp": 32,
                    "duration_embed_dim": 16}
    }
    parser = argparse.ArgumentParser(description="Run MobTCast Model Training and Evaluation.")
    parser.add_argument('--dataset', type=str, default='fsq', choices=['fsq', 'geolife'], help="The dataset to use.")
    parser.add_argument('--city', type=str, default='nyc', choices=['nyc', 'tky'], help="City for the Foursquare (fsq) dataset.")
    parser.add_argument('--previous_days', type=int, default=7, help="Number of previous days to consider for history.")
    parser.add_argument('--hidden_size', type=int, default=256, help="Hidden size of the model.")
    parser.add_argument('--obs_len', type=int, default=20, help="Observation length (sequence length).")
    parser.add_argument('--m1_transformer_nheads', type=int, default=8, help="Number of heads in the Transformer encoder.")
    parser.add_argument('--m1_transformer_nlayers', type=int, default=4, help="Number of layers in the Transformer encoder.")
    parser.add_argument('--m1_transformer_dim_feedforward', type=int, default=1024, help="Dimension of the feedforward network.")
    parser.add_argument('--learning_rate', type=float, default=2e-4, help="Learning rate.")
    parser.add_argument('--epochs', type=int, default=80, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size.")
    parser.add_argument('--dropout_rate', type=float, default=0.4, help="Dropout rate.")
    parser.add_argument('--patience', type=int, default=15, help="Patience for early stopping.")
    parser.add_argument('--gradient_clip_value', type=float, default=1.0, help="Gradient clipping value (0 for no clipping).")
    parser.add_argument('--m1_alpha_loc_loss', type=float, default=0.005, help="Weight for the MSE location auxiliary loss.")
    parser.add_argument('--m1_beta_loc_loss', type=float, default=0.002, help="Weight for the L1 location auxiliary loss.")
    parser.add_argument('--lr_scheduler_type', type=str, default='cosine_annealing_warmup', choices=['reduce_on_plateau', 'cosine_annealing_warmup'], help="Type of learning rate scheduler.")
    parser.add_argument('--lr_warmup_epochs', type=int, default=10, help="Number of warmup epochs for the scheduler.")
    parser.add_argument('--m1_use_real_poi_coords', action='store_true', default=True, help="Use real POI coordinates as features.")
    parser.add_argument('--m1_normalize_coords', action='store_true', default=True, help="Normalize POI coordinates if used.")
    parser.add_argument('--m1_use_poi_categories', action='store_true', default=True, help="Use POI categories as semantic features.")
    parser.add_argument('--poi_category_column_name', type=str, default='poi_category', help="Column name for POI categories in the CSV.")
    parser.add_argument('--topk_acc', type=str, default="1,5,10", help="Comma-separated k values for accuracy calculation.")
    parser.add_argument('--ndcg_k_val', type=int, default=10, help="k value for NDCG calculation.")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use.")
    parser.add_argument('--seed', type=int, default=2026, help="Random seed.")
    parser.add_argument('--num_workers', type=int, default=0, help="Number of workers for DataLoader.")
    parser.add_argument('--resume_training', action='store_true', help="Resume training from a checkpoint.")
    parser.add_argument('--force_regenerate_data', action='store_true', help="Force regeneration of preprocessed data files.")
    args = parser.parse_args()

    # --- SHARED SETUP ---
    setup_seed(args.seed)
    acc_topk_values = [int(k_str) for k_str in args.topk_acc.split(',')]; max_k_for_prediction = max(max(acc_topk_values), args.ndcg_k_val)

    # =========================================================================================
    # CONDITIONAL EXECUTION: FSQ vs GEOLIFE
    # =========================================================================================

    if args.dataset == 'fsq':
        # --- FSQ EXECUTION PATH ---
        print("\n" + "="*50)
        print("执行 FSQ 数据集模式...")
        print("="*50 + "\n")

        # 1. Override args with FSQ-specific hyperparameters
        print("正在为 FSQ 应用特定的超参数...")
        args.embed_dims = {'loc': 128, 'user': DATASET_CONFIGS['fsq']['user_embed_dim']}
        args.hidden_size = 256
        args.dropout_rate = 0.3
        args.obs_len = 20
        args.learning_rate = 0.001
        args.weight_decay = 1e-5
        args.epochs = 150
        args.batch_size = 64
        args.patience = 15
        args.lr_scheduler_patience = 7
        args.lr_scheduler_factor = 0.1
        args.gradient_clip_value = 1.0
        args.m1_use_real_poi_coords = False
        args.m1_use_poi_categories = False
        args.use_duration_feature = False
        args.m1_weekday_category_num = 7
        args.m1_tid_num = 48

        # 2. Data Loading Setup
        dataset_name_for_loader = f"fsq_{args.city}"
        source_filename_prefix = f"foursquare_{args.city}"
        data_root_dir = os.path.join(project_root_dir, 'data', 'fsq', args.city)
        args.model_type_for_dataloader = f"mobtcast_{dataset_name_for_loader}_fsq_model"

        # 3. Instantiate Datasets and Dataloaders
        train_dataset = sp_loc_dataset(source_root=data_root_dir, dataset=dataset_name_for_loader, data_type="train", previous_day=args.previous_days, model_type=args.model_type_for_dataloader, args_ref=args, source_filename_prefix=source_filename_prefix)
        loc_vocab_size, num_users, _ = get_max_user_id_and_loc_vocab_size(train_dataset, data_root_dir, source_filename_prefix, DATASET_CONFIGS['fsq']['default_num_users'])
        eval_dataset = sp_loc_dataset(source_root=data_root_dir, dataset=dataset_name_for_loader, data_type="validation", previous_day=args.previous_days, model_type=args.model_type_for_dataloader, args_ref=args, source_filename_prefix=source_filename_prefix)
        test_dataset = sp_loc_dataset(source_root=data_root_dir, dataset=dataset_name_for_loader, data_type="test", previous_day=args.previous_days, model_type=args.model_type_for_dataloader, args_ref=args, source_filename_prefix=source_filename_prefix)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: custom_collate_fn_with_padding_fsq(b, args.obs_len, local_collate_fn_fsq), num_workers=args.num_workers)
        eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: custom_collate_fn_with_padding_fsq(b, args.obs_len, local_collate_fn_fsq), num_workers=args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: custom_collate_fn_with_padding_fsq(b, args.obs_len, local_collate_fn_fsq), num_workers=args.num_workers)
        print(f"数据集大小: 训练集={len(train_dataset)}, 验证集={len(eval_dataset)}, 测试集={len(test_dataset)}")

        # 4. Instantiate Predictor and Run
        predictor = MobTCast_Predictor_fsq(loc_vocab_size=loc_vocab_size, num_users=num_users, user_args_instance=args)
        checkpoint_dir = f"./checkpoints_mobtcast_fsq/{dataset_name_for_loader}/"
        checkpoint_file_path = os.path.join(checkpoint_dir, "fsq_model_ckpt.pth")
        
        if args.resume_training: predictor.load_checkpoint(checkpoint_file_path, load_best=False)
        
        predictor.train(train_loader, eval_loader, checkpoint_file_path=checkpoint_file_path)
        predictor.load_checkpoint(checkpoint_file_path, load_best=True)
        all_preds_ids, all_targets_ids = predictor.predict(test_loader, topk=max_k_for_prediction)
        
        # 5. Evaluate
        print(f"\nMobTCast (FSQ Model) 评估 (dataset={dataset_name_for_loader}):")
        all_metrics_results = calculate_all_metrics(all_preds_ids, all_targets_ids, acc_topk_values, args.ndcg_k_val)
        for k_val in acc_topk_values: print(f"  Accuracy@{k_val}:  {all_metrics_results.get(f'Acc@{k_val}', 0.0):.4f}")
        print(f"  F1-score (Top-1): {all_metrics_results.get('F1', 0.0):.4f}")
        print(f"  Recall (Top-1):   {all_metrics_results.get('Recall', 0.0):.4f}")
        print(f"  NDCG@{args.ndcg_k_val}:       {all_metrics_results.get(f'NDCG@{args.ndcg_k_val}', 0.0):.4f}")
        print(f"  MRR:              {all_metrics_results.get('MRR', 0.0):.4f}")


    else: # This is the GEOLIFE EXECUTION PATH
        print("\n" + "="*50)
        print("执行 GEOLIFE 数据集模式...")
        print("="*50 + "\n")
        
        # ==================================================================
        # START: MODIFICATION BLOCK FOR GEOLIFE PARAMS
        # Parameters are now aligned with mobtcast2.py's geolife settings
        # ==================================================================

        # 1. Configure args for GEOLIFE to match mobtcast2.py
        args.seed = 2025  # Set seed to match mobtcast2
        setup_seed(args.seed) # Re-seed with the new value

        args.use_duration_feature = False  # Disabled as it's not used in mobtcast2's simple model
        args.m1_use_poi_categories = False # Disabled to match mobtcast2
        
        # Enable coordinate auxiliary loss as requested
        args.m1_use_real_poi_coords = True
        args.m1_normalize_coords = True # Good practice when using coords
        args.m1_alpha_loc_loss = 0.1
        args.m1_beta_loc_loss = 0.05
        
        # Align embedding dimensions with mobtcast2
        cfg = DATASET_CONFIGS['geolife']
        args.embed_dims = {'user': 16, 'loc': 64} # From mobtcast2 config
        # These are calculated inside the model in mobtcast2, so we don't need to set them as args
        args.m1_semantic_embed_dim = 96 
        args.m1_tid_embed_dim = 96
        args.m1_coord_embedding_dim = 32 # Default value
        
        # Align model architecture and training hyperparameters with mobtcast2
        args.hidden_size = 256
        args.dropout_rate = 0.3
        args.m1_transformer_nlayers = 3
        args.m1_transformer_nheads = 8
        args.m1_transformer_dim_feedforward = 512
        
        # Align training process parameters with mobtcast2
        args.learning_rate = 5e-4
        args.weight_decay = 1e-5
        args.epochs = 100
        args.batch_size = 128
        args.patience = 15
        
        # Align scheduler parameters with mobtcast2's simple predictor
        args.lr_scheduler_type = 'reduce_on_plateau'
        args.lr_scheduler_patience = 5
        args.lr_scheduler_factor = 0.2
        
        # Other parameters from mobtcast2
        args.adam_beta1, args.adam_beta2, args.adam_eps = 0.9, 0.999, 1e-8 # Default AdamW values used in mobtcast2's simple predictor
        args.gradient_accumulation_steps = 1
        
        # General vocab sizes (will be overridden by data scan if needed)
        args.m1_poi_category_vocab_size = 100 
        args.m1_weekday_category_num = 7
        args.m1_tid_num = 48

        print("Geolife 参数已修改，与 mobtcast2.py 的设置对齐，并已启用坐标辅助损失。")
        # ==================================================================
        # END: MODIFICATION BLOCK
        # ==================================================================
        
        # 2. Data Loading Setup
        data_root_dir = os.path.join(project_root_dir, 'data', args.dataset)
        dataset_name_for_loader = args.dataset
        source_filename_prefix = args.dataset
        args.model_type_for_dataloader = f"mobtcast_{dataset_name_for_loader}_opt_pd{args.previous_days}_dur{args.use_duration_feature}"
        args.checkpoint_dir = f"./checkpoints_mobtcast_optimized/{dataset_name_for_loader}/obs{args.obs_len}_hid{args.hidden_size}_dur{args.use_duration_feature}"
        args.checkpoint_name = "geolife_model_ckpt.pth"
        checkpoint_file_path = os.path.join(args.checkpoint_dir, args.checkpoint_name)
        if not os.path.exists(args.checkpoint_dir): os.makedirs(args.checkpoint_dir, exist_ok=True)

        # 3. Instantiate Datasets and Dataloaders
        train_dataset = sp_loc_dataset(source_root=data_root_dir, dataset=dataset_name_for_loader, data_type="train", previous_day=args.previous_days, model_type=args.model_type_for_dataloader, args_ref=args, source_filename_prefix=source_filename_prefix)
        loc_vocab_size, num_users, _ = get_max_user_id_and_loc_vocab_size(train_dataset, data_root_dir, source_filename_prefix, cfg['default_num_users'])
        eval_dataset = sp_loc_dataset(source_root=data_root_dir, dataset=dataset_name_for_loader, data_type="validation", previous_day=args.previous_days, model_type=args.model_type_for_dataloader, args_ref=args, source_filename_prefix=source_filename_prefix)
        test_dataset = sp_loc_dataset(source_root=data_root_dir, dataset=dataset_name_for_loader, data_type="test", previous_day=args.previous_days, model_type=args.model_type_for_dataloader, args_ref=args, source_filename_prefix=source_filename_prefix)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: custom_collate_fn_with_padding_geolife(b, args.obs_len, local_collate_fn_geolife, args.m1_use_real_poi_coords), num_workers=args.num_workers, drop_last=True)
        eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: custom_collate_fn_with_padding_geolife(b, args.obs_len, local_collate_fn_geolife, args.m1_use_real_poi_coords), num_workers=args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: custom_collate_fn_with_padding_geolife(b, args.obs_len, local_collate_fn_geolife, args.m1_use_real_poi_coords), num_workers=args.num_workers)
        print(f"数据集大小: 训练集={len(train_dataset)}, 验证集={len(eval_dataset)}, 测试集={len(test_dataset)}")
        
        # 4. Instantiate Predictor and Run
        predictor = MobTCast_Predictor_geolife(loc_vocab_size=loc_vocab_size, num_users=num_users, user_args_instance=args)
        if args.resume_training: predictor.load_checkpoint(checkpoint_file_path, load_best=False)

        predictor.train(train_loader, eval_loader, checkpoint_file_path=checkpoint_file_path)
        predictor.load_checkpoint(checkpoint_file_path, load_best=True)
        all_preds_ids, all_targets_ids = predictor.predict(test_loader, topk=max_k_for_prediction)
        
        # 5. Evaluate
        print(f"\nMobTCast (Geolife Model) 评估 (dataset={dataset_name_for_loader}):")
        all_metrics_results = calculate_all_metrics(all_preds_ids, all_targets_ids, acc_topk_values, args.ndcg_k_val)
        for k_val in acc_topk_values: print(f"  Accuracy@{k_val}:  {all_metrics_results.get(f'Acc@{k_val}', 0.0):.4f}")
        print(f"  F1-score (Top-1): {all_metrics_results.get('F1', 0.0):.4f}")
        print(f"  Recall (Top-1):   {all_metrics_results.get('Recall', 0.0):.4f}")
        print(f"  NDCG@{args.ndcg_k_val}:       {all_metrics_results.get(f'NDCG@{args.ndcg_k_val}', 0.0):.4f}")
        print(f"  MRR:              {all_metrics_results.get('MRR', 0.0):.4f}")