import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import sys
import os
import json
from sklearn.metrics import f1_score, recall_score
import pandas as pd
import argparse

try:
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_dir = os.path.abspath(os.path.join(current_script_dir, '..'))
    if project_root_dir not in sys.path:
        sys.path.insert(0, project_root_dir)
    from utils.dataloader import sp_loc_dataset, collate_fn 
    from utils.utils import setup_seed 
except ImportError as e:
    print(f"导入模块错误: {e}")
    sys.exit(1)

class TargetAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.scale = np.sqrt(hidden_dim)

    def forward(self, lstm_out, query_vector):
        query = self.query_proj(query_vector).unsqueeze(1)
        keys = self.key_proj(lstm_out)
        scores = torch.bmm(query, keys.transpose(1, 2)) / self.scale
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights, lstm_out)
        return context.squeeze(1), attn_weights.squeeze(1)


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dims, hidden_size, num_layers, dropout_rate, device, 
                 num_users, max_time_diff, bidirectional=False, use_attention=False):
        super(LSTMModel, self).__init__()
        self.device = device
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.embed_dims = embed_dims

        self.loc_embedding = nn.Embedding(vocab_size, embed_dims['loc'], padding_idx=0)
        self.time_embedding = nn.Embedding(48, embed_dims['time'], padding_idx=0)
        self.weekday_embedding = nn.Embedding(7, embed_dims['weekday'], padding_idx=0)
        self.diff_embedding = nn.Embedding(max_time_diff + 1, embed_dims['diff'], padding_idx=0)
        self.user_embedding = nn.Embedding(num_users, embed_dims['user'])

        lstm_input_dim = embed_dims['loc'] + embed_dims['time'] + embed_dims['weekday'] + embed_dims['diff']
        
        self.lstm = nn.LSTM(
            lstm_input_dim, hidden_size, num_layers,
            batch_first=True, dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        self.lstm_output_dim = hidden_size * 2 if bidirectional else hidden_size

        if use_attention:
            self.attention = TargetAttention(self.lstm_output_dim)
            self.fc_input_dim = self.lstm_output_dim * 2 + embed_dims['user'] 
        else:
            self.fc_input_dim = self.lstm_output_dim + embed_dims['user']

        self.dropout_layer = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(self.fc_input_dim, vocab_size)

    def forward(self, x_loc, x_time, x_weekday, x_diff, x_user, h_0=None, c_0=None):
        loc_embed = self.loc_embedding(x_loc)
        time_embed = self.time_embedding(x_time)
        weekday_embed = self.weekday_embedding(x_weekday)
        diff_embed = self.diff_embedding(x_diff)
        
        # --- USER ID 修正 ---
        # x_user 是一个序列，我们只取第一个时间步的 user_id (因为它们都相同)
        user_id_tensor = x_user[:, 0]
        user_embed = self.user_embedding(user_id_tensor)

        combined_embed = torch.cat((loc_embed, time_embed, weekday_embed, diff_embed), dim=2)
        combined_embed = self.dropout_layer(combined_embed)

        batch_size = x_loc.size(0)
        num_directions = 2 if self.bidirectional else 1
        if h_0 is None or c_0 is None:
            h_0 = torch.zeros(self.lstm.num_layers * num_directions, batch_size, self.lstm.hidden_size).to(self.device)
            c_0 = torch.zeros(self.lstm.num_layers * num_directions, batch_size, self.lstm.hidden_size).to(self.device)

        lstm_out, (hidden, cell) = self.lstm(combined_embed, (h_0, c_0))
        
        if self.use_attention:
            query = lstm_out[:, -1, :]
            attn_context, _ = self.attention(lstm_out, query)
            processed_representation = torch.cat([query, attn_context], dim=1)
        else:
            processed_representation = lstm_out[:, -1, :]

        processed_representation = self.dropout_layer(F.relu(processed_representation)) 
        final_features = torch.cat([processed_representation, user_embed], dim=1)
        out = self.fc(final_features)
        return out, hidden, cell


class LSTM_Predictor:
    def __init__(self, loc_vocab_size, embed_dims, hidden_size, num_layers, dropout_rate,
                 learning_rate, epochs, device, num_users, max_time_diff, patience=10, 
                 bidirectional_lstm=False, use_attention=False, 
                 lr_scheduler_patience=2, lr_scheduler_factor=0.5, gradient_clip_value=1.0,
                 weight_decay=1e-5): 
        
        self.loc_vocab_size = loc_vocab_size 
        self.embed_dims = embed_dims
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = device
        self.patience = patience 
        self.gradient_clip_value = gradient_clip_value
        self.num_users = num_users
        self.max_time_diff = max_time_diff 

        self.model = LSTMModel(
            self.loc_vocab_size, self.embed_dims, self.hidden_size,
            self.num_layers, self.dropout_rate, self.device,
            self.num_users, self.max_time_diff, 
            bidirectional=bidirectional_lstm,
            use_attention=use_attention 
        ).to(self.device)
        
        self.criterion = nn.CrossEntropyLoss() 
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=weight_decay) 
        
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=lr_scheduler_factor, 
            patience=lr_scheduler_patience
        )

        self.start_epoch = 0
        self.best_val_loss = float('inf')
        print(f"LSTM_Predictor 初始化: loc_vocab_size={loc_vocab_size}, embed_dims={embed_dims}, "
              f"hidden_size={hidden_size}, num_layers={num_layers}, dropout={dropout_rate}, "
              f"lr={learning_rate}, weight_decay={weight_decay}, epochs={epochs}, bidirectional={bidirectional_lstm}, attention={use_attention}, " 
              f"num_users={num_users}, max_time_diff_param_for_model={max_time_diff} (Embedding size will be this + 1)")
        print(f"模型将运行在: {self.device}")

    def save_checkpoint(self, epoch, checkpoint_file_path, is_best=False):
        checkpoint = {
            'epoch': epoch, 'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
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
            else: print(f"未找到最佳检查点 {best_path}，尝试加载常规检查点。")
        
        if not os.path.exists(actual_path_to_load):
            print(f"检查点文件 {actual_path_to_load} 未找到。")
            return False
        try:
            print(f"正在从 {actual_path_to_load} 加载检查点...")
            checkpoint = torch.load(actual_path_to_load, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                 self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if not load_best:
                self.start_epoch = checkpoint['epoch'] + 1 
                self.best_val_loss = checkpoint['best_val_loss']
                print(f"恢复训练: Epoch {self.start_epoch}, Best Val Loss: {self.best_val_loss:.4f}")
            else:
                print(f"最佳模型加载成功 (Epoch {checkpoint['epoch']})。")
            return True
        except Exception as e:
            print(f"加载检查点 {actual_path_to_load} 失败: {e}。")
            return False

    def train(self, train_loader, eval_loader=None, checkpoint_file_path="checkpoint.pth"):
        epochs_no_improve_early_stopping = 0
        checkpoint_dir = os.path.dirname(checkpoint_file_path)
        if checkpoint_dir and not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)

        for epoch in range(self.start_epoch, self.epochs):
            self.model.train()
            total_loss = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [训练中]", unit="batch")
            for x_loc_batch, y_batch, x_dict_batch in pbar: 
                # --- TRANSPOSE 修正 ---
                # collate_fn 现在直接返回 batch_first=True, 不再需要 transpose
                x_loc_batch = x_loc_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                x_time_batch = x_dict_batch['time'].to(self.device)
                x_weekday_batch = x_dict_batch['weekday'].to(self.device)
                x_diff_batch = x_dict_batch['diff'].to(self.device)
                x_user_batch = x_dict_batch['user'].to(self.device)

                self.optimizer.zero_grad()
                output, _, _ = self.model(
                    x_loc_batch, x_time_batch, x_weekday_batch, 
                    x_diff_batch, x_user_batch
                )
                loss = self.criterion(output, y_batch)
                loss.backward()
                if self.gradient_clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_value)
                self.optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{self.optimizer.param_groups[0]['lr']:.1e}")

            avg_train_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{self.epochs}, Avg Train Loss: {avg_train_loss:.4f}, LR: {current_lr:.1e}")

            if eval_loader:
                self.model.eval()
                total_eval_loss = 0
                eval_pbar = tqdm(eval_loader, desc=f"Epoch {epoch+1}/{self.epochs} [验证中]", unit="batch")
                with torch.no_grad():
                    for x_loc_batch, y_batch, x_dict_batch in eval_pbar:
                        x_loc_batch = x_loc_batch.to(self.device)
                        y_batch = y_batch.to(self.device)
                        x_time_batch = x_dict_batch['time'].to(self.device)
                        x_weekday_batch = x_dict_batch['weekday'].to(self.device)
                        x_diff_batch = x_dict_batch['diff'].to(self.device)
                        x_user_batch = x_dict_batch['user'].to(self.device)
                        
                        output, _, _ = self.model(
                            x_loc_batch, x_time_batch, x_weekday_batch, 
                            x_diff_batch, x_user_batch
                        )
                        loss = self.criterion(output, y_batch)
                        total_eval_loss += loss.item()
                        eval_pbar.set_postfix(loss=f"{loss.item():.4f}")
                
                avg_eval_loss = total_eval_loss / len(eval_loader) if len(eval_loader) > 0 else float('inf')
                print(f"Epoch {epoch+1}/{self.epochs}, Validation Loss: {avg_eval_loss:.4f}")
                self.scheduler.step(avg_eval_loss)

                is_best = avg_eval_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = avg_eval_loss
                    epochs_no_improve_early_stopping = 0
                    print(f"Validation loss improved to {self.best_val_loss:.4f}.")
                    self.save_checkpoint(epoch, checkpoint_file_path, is_best=True)
                else:
                    epochs_no_improve_early_stopping += 1
                    print(f"Validation loss did not improve for {epochs_no_improve_early_stopping} epochs.")
                    if epoch % 5 == 0 or epoch == self.epochs -1 :
                         self.save_checkpoint(epoch, checkpoint_file_path, is_best=False)
                if epochs_no_improve_early_stopping >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch+1}. Best val loss: {self.best_val_loss:.4f}")
                    break
            else: 
                if epoch % 5 == 0 or epoch == self.epochs -1 :
                    self.save_checkpoint(epoch, checkpoint_file_path, is_best=False)
        print("Training finished.")

    def predict(self, test_loader, topk=1):
        self.model.eval()
        all_preds = []
        all_targets = []
        print("Starting prediction...")
        with torch.no_grad():
            for x_loc_batch, y_batch, x_dict_batch in tqdm(test_loader, desc="Predicting", unit="batch"):
                x_loc_batch = x_loc_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                x_time_batch = x_dict_batch['time'].to(self.device)
                x_weekday_batch = x_dict_batch['weekday'].to(self.device)
                x_diff_batch = x_dict_batch['diff'].to(self.device)
                x_user_batch = x_dict_batch['user'].to(self.device)

                output, _, _ = self.model(
                    x_loc_batch, x_time_batch, x_weekday_batch, 
                    x_diff_batch, x_user_batch
                )
                probabilities = torch.softmax(output, dim=1)
                _, top_indices = torch.topk(probabilities, topk, dim=1)
                all_preds.extend(top_indices.cpu().tolist())
                all_targets.extend(y_batch.cpu().tolist())
        if not all_targets: print("Warning: No predictions were made.")
        return all_preds, all_targets

def calculate_all_metrics(all_preds, all_targets, acc_topk_list, ndcg_k=10):
    metrics_results = {}
    num_samples = len(all_targets)
    if num_samples == 0:
        for k_val in acc_topk_list: metrics_results[f'Acc@{k_val}'] = 0.0
        metrics_results.update({'F1': 0.0, 'Recall': 0.0, f'NDCG@{ndcg_k}': 0.0, 'MRR': 0.0})
        return metrics_results
    correct_counts = {k: 0 for k in acc_topk_list}; ndcg_scores, rr_scores = [], []
    top1_preds = [p[0] if p else -1 for p in all_preds]
    for i in range(num_samples):
        actual, predicted = all_targets[i], all_preds[i]
        for k_val in acc_topk_list:
            if actual in predicted[:k_val]: correct_counts[k_val] += 1
        preds_ndcg = predicted[:min(len(predicted), ndcg_k)]
        try: ndcg_scores.append(1.0 / np.log2(preds_ndcg.index(actual) + 2))
        except ValueError: ndcg_scores.append(0.0)
        try: rr_scores.append(1.0 / (predicted.index(actual) + 1))
        except ValueError: rr_scores.append(0.0)
    for k_val in acc_topk_list: metrics_results[f'Acc@{k_val}'] = (correct_counts[k_val]/num_samples) if num_samples else 0.0
    metrics_results.update({
        'F1': f1_score(all_targets, top1_preds, average="weighted", zero_division=0),
        'Recall': recall_score(all_targets, top1_preds, average="weighted", zero_division=0),
        f'NDCG@{ndcg_k}': np.mean(ndcg_scores) if ndcg_scores else 0.0,
        'MRR': np.mean(rr_scores) if rr_scores else 0.0
    })
    return metrics_results

def get_max_user_id_and_loc_vocab_size(dataset_instance, data_root, dataset_name, city, default_num_users_config):
    print("正在确定位置词汇表大小和最大用户ID...")
    max_loc_id = 0
    default_num_users = default_num_users_config
    try:
        if dataset_name == 'fsq':
            if city is None: raise ValueError("必须为 fsq 数据集指定城市")
            ori_data_path = os.path.join(data_root, dataset_name, city, f"dataSet_foursquare_{city}.csv")
        else:
            ori_data_path = os.path.join(data_root, dataset_name, f"dataSet_{dataset_name}.csv")
        
        if os.path.exists(ori_data_path):
            ori_df = pd.read_csv(ori_data_path) 
            max_user_id_from_file = ori_df['user_id'].max()
            num_users = int(max_user_id_from_file) + 1 
            print(f"从文件 {os.path.basename(ori_data_path)} 推断的最大用户ID: {max_user_id_from_file}, 用户数: {num_users}")
        else:
            print(f"警告: 原始数据集文件 {ori_data_path} 未找到。将使用默认值 {default_num_users}。")
            num_users = default_num_users 
    except Exception as e:
        print(f"读取原始数据集文件以获取用户数时出错: {e}。将使用默认值 {default_num_users}。")
        num_users = default_num_users

    if not dataset_instance.data: 
        try: _ = dataset_instance[0] 
        except Exception as e: print(f"访问数据集元素时出错: {e}"); return -1, num_users
    if not dataset_instance.data: return -1, num_users

    max_user_id_in_data = 0
    for record in tqdm(dataset_instance.data, desc="扫描数据记录 (位置和用户ID)"):
        if 'X' in record and record['X'] is not None and len(record['X']) > 0:
            current_max_x = np.max(record['X'])
            if current_max_x > max_loc_id: max_loc_id = current_max_x
        if 'Y' in record and record['Y'] is not None:
            if record['Y'] > max_loc_id: max_loc_id = record['Y']
        if 'user_X' in record and record['user_X'] is not None and len(record['user_X']) > 0:
             current_user_id = record['user_X'][0]
             if current_user_id > max_user_id_in_data: max_user_id_in_data = current_user_id

    if num_users <= max_user_id_in_data :
        print(f"警告：数据中的最大用户ID ({max_user_id_in_data}) >= 从文件/默认推断的用户数 ({num_users-1})。将使用 {max_user_id_in_data + 1}。")
        num_users = int(max_user_id_in_data) + 1

    loc_vocab_size = int(max_loc_id) + 1
    print(f"从数据推断的最大位置ID: {max_loc_id}。位置词汇表大小: {loc_vocab_size}")
    print(f"最终用于用户嵌入层的用户数量: {num_users}")
    return loc_vocab_size, num_users

# --- 主执行模块 ---
if __name__ == '__main__':
    DATASET_CONFIGS = {
        "fsq": {
            "tky": {"default_num_users": 2500, "user_embed_dim": 32},
            "nyc": {"default_num_users": 1000, "user_embed_dim": 32}
        },
        "geolife": {"default_num_users": 200, "user_embed_dim": 16}
    }
    parser = argparse.ArgumentParser(description="LSTM-based Location Prediction Model")
    parser.add_argument('--dataset', type=str, required=True, choices=['fsq', 'geolife'], help='要使用的数据集')
    parser.add_argument('--city', type=str, choices=['tky', 'nyc'], help='当数据集是 fsq 时，必须指定城市')
    args_cmd = parser.parse_args()

    class Args:
        dataset = args_cmd.dataset
        city = args_cmd.city
        embed_dims = {'loc': 64, 'time': 16, 'weekday': 16, 'diff': 16, 'user': 32}
        hidden_size = 128      
        num_layers = 2         
        dropout_rate = 0.3    
        bidirectional_lstm = False 
        use_attention = True
        learning_rate = 0.001 
        weight_decay = 1e-5   
        epochs = 150          
        batch_size = 64       
        patience = 20         
        lr_scheduler_patience = 7 
        lr_scheduler_factor = 0.1 
        gradient_clip_value = 1.0 
        previous_days = 7     
        topk_acc = "1,5,10" 
        ndcg_k_val = 10     
        device = "cuda" if torch.cuda.is_available() else "cpu"
        seed = 2023
        num_workers = 0 
        resume_training = False 
    args = Args()

    if args.dataset == 'fsq' and not args.city:
        raise ValueError("错误：当数据集为 'fsq' 时，必须通过 --city 参数指定城市 (tky 或 nyc)。")

    if args.dataset == 'fsq':
        dataset_specific_config = DATASET_CONFIGS[args.dataset][args.city]
    else:
        dataset_specific_config = DATASET_CONFIGS[args.dataset]
    
    args.embed_dims['user'] = dataset_specific_config.get('user_embed_dim', args.embed_dims['user'])
    default_num_users_for_get_func = dataset_specific_config.get('default_num_users', 1000)

    city_suffix = f"_{args.city}" if args.dataset == 'fsq' else ""
    args.model_type_for_dataloader = f"lstm_target_attn_{args.dataset}{city_suffix}_pd{args.previous_days}"
    args.checkpoint_dir = f"./checkpoints_lstm_target_attn/{args.dataset}{city_suffix}/pd{args.previous_days}"
    args.checkpoint_name_template = "lstm_ds-{}{}_bidir-{}_attn-{}_h-{}_l-{}_d-{}_pd-{}_ue-{}_checkpoint.pth"
    args.checkpoint_name = args.checkpoint_name_template.format(
        args.dataset, city_suffix, args.bidirectional_lstm, args.use_attention, 
        args.hidden_size, args.num_layers, args.dropout_rate, args.previous_days, 
        args.embed_dims['user']
    )
    acc_topk_values = [int(k_str) for k_str in args.topk_acc.split(',')]
    max_k_for_prediction = max(max(acc_topk_values), args.ndcg_k_val)

    print(f"--- 参数设置 (Dataset: {args.dataset}, City: {args.city or 'N/A'}) ---")
    for k, v in vars(args).items():
        if k != 'checkpoint_name_template': print(f"{k}: {v}")
    print("-----------------------------------------------------------------")
    print(f"使用设备: {args.device}")
    setup_seed(args.seed) 
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_file_path = os.path.join(args.checkpoint_dir, args.checkpoint_name)
    print(f"检查点将保存至/从: {checkpoint_file_path}")
    data_root_dir = os.path.join(project_root_dir, 'data')
    print(f"数据根目录: {data_root_dir}")

    try:
        print("加载训练数据...")
        train_dataset = sp_loc_dataset(
            source_root=data_root_dir, dataset=args.dataset, city=args.city, data_type="train",
            previous_day=args.previous_days, model_type=args.model_type_for_dataloader
        )
        loc_vocab_size, num_users = get_max_user_id_and_loc_vocab_size(
            train_dataset, data_root_dir, args.dataset, args.city, default_num_users_for_get_func
        )
        if loc_vocab_size <= 0 or num_users <=0 : sys.exit(1)
        print("加载验证数据...")
        eval_dataset = sp_loc_dataset(
            source_root=data_root_dir, dataset=args.dataset, city=args.city, data_type="validation", 
            previous_day=args.previous_days, model_type=args.model_type_for_dataloader
        )
        print("加载测试数据...")
        test_dataset = sp_loc_dataset(
            source_root=data_root_dir, dataset=args.dataset, city=args.city, data_type="test",
            previous_day=args.previous_days, model_type=args.model_type_for_dataloader
        )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers)
        eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)
        print(f"训练集: {len(train_dataset)}, 验证集: {len(eval_dataset)}, 测试集: {len(test_dataset)}")
        if len(train_dataset) == 0: sys.exit(1)
    except FileNotFoundError as e: print(f"加载数据错误: {e}"); sys.exit(1)
    except Exception as e:
        print(f"发生未知错误于数据加载: {e}"); import traceback; traceback.print_exc(); sys.exit(1)

    predictor = LSTM_Predictor(
        loc_vocab_size=loc_vocab_size, embed_dims=args.embed_dims, hidden_size=args.hidden_size,
        num_layers=args.num_layers, dropout_rate=args.dropout_rate, learning_rate=args.learning_rate,
        epochs=args.epochs, device=args.device, patience=args.patience,
        num_users=num_users, max_time_diff=args.previous_days, 
        bidirectional_lstm=args.bidirectional_lstm, 
        use_attention=args.use_attention, 
        lr_scheduler_patience=args.lr_scheduler_patience, lr_scheduler_factor=args.lr_scheduler_factor,
        gradient_clip_value=args.gradient_clip_value,
        weight_decay=args.weight_decay 
    )

    if args.resume_training:
        predictor.load_checkpoint(checkpoint_file_path, load_best=False) 
    
    print("开始训练...")
    predictor.train(train_loader, eval_loader, checkpoint_file_path=checkpoint_file_path) 
    print(f"训练结束。正在从最佳检查点加载模型进行最终评估...")
    load_success = predictor.load_checkpoint(checkpoint_file_path, load_best=True) 
    if not load_success :
         print("警告：未能加载最佳检查点。将使用当前模型状态评估。")
    print("开始预测和评估...")
    all_preds, all_targets = predictor.predict(test_loader, topk=max_k_for_prediction)

    if not all_targets: print("没有可用于评估的目标。退出。")
    else:
        print(f"\nLSTM 模型评估结果 (dataset={args.dataset}, city={args.city or 'N/A'}, attention={args.use_attention}):") 
        all_metrics_results = calculate_all_metrics(all_preds, all_targets, acc_topk_values, args.ndcg_k_val)
        for k_val in acc_topk_values: print(f"  Accuracy@{k_val}:  {all_metrics_results.get(f'Acc@{k_val}', 0.0):.4f}")
        print(f"  F1-score (Top-1): {all_metrics_results.get('F1', 0.0):.4f}")
        print(f"  Recall (Top-1):   {all_metrics_results.get('Recall', 0.0):.4f}")
        print(f"  NDCG@{args.ndcg_k_val}:       {all_metrics_results.get(f'NDCG@{args.ndcg_k_val}', 0.0):.4f}")
        print(f"  MRR:              {all_metrics_results.get('MRR', 0.0):.4f}")