import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

# 参数设置
DATASET_PATH = 'D:/code/LLM-Mob/data/fsq/tky/dataset_foursquare_tky.csv'  # 替换为文件路径
OUTPUT_DIR = 'D:/code/LLM-Mob/data/fsq'  # 对应代码中dataname='fsq'
NUM_CONTEXT_STAY = 5  # 上下文记录数（与代码中num_context_stay=5一致）

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. 读取并预处理数据
df = pd.read_csv(DATASET_PATH)
# 按用户和时间排序（保持时间序列连续性）
df = df.sort_values(['user_id', 'start_day', 'start_min'])

# 2. 分层划分训练/验证/测试集（按用户分组）
train_data_list, valid_data_list, test_samples = [], [], []

for uid, user_data in df.groupby('user_id'):
    # 按时间排序用户数据
    user_data = user_data.sort_values(['start_day', 'start_min'])
    
    # 划分非测试集(80%)和测试集(20%)
    non_test, test = train_test_split(user_data, test_size=0.2, shuffle=False)
    
    # 从非测试集中划分训练集(87.5%)和验证集(12.5%)
    train, valid = train_test_split(non_test, test_size=0.125, shuffle=False)
    
    # 收集划分结果
    train_data_list.append(train)
    valid_data_list.append(valid)
    
    # 3. 生成测试集样本（滑动窗口）
    test_records = test.to_dict('records')
    for i in range(len(test_records) - NUM_CONTEXT_STAY):
        context = test_records[i:i+NUM_CONTEXT_STAY]
        target = test_records[i+NUM_CONTEXT_STAY]
        
        test_samples.append({
            'user_X': uid,
            'start_min_X': [r['start_min'] for r in context],
            'weekday_X': [r['weekday'] for r in context],
            'X': [r['location_id'] for r in context],  # 上下文位置ID
            'start_min_Y': target['start_min'],        # 目标时间
            'weekday_Y': target['weekday'],            # 目标星期
            'Y': target['location_id']                # 目标位置ID
        })

# 4. 合并并保存数据集
## 训练集和验证集（保持原始列结构）
train_data = pd.concat(train_data_list)
valid_data = pd.concat(valid_data_list)
train_data.to_csv(f'{OUTPUT_DIR}/fsq_train.csv', index=False, 
                  columns=['id','user_id','location_id','latitude','longitude','start_day','start_min','weekday'])
valid_data.to_csv(f'{OUTPUT_DIR}/fsq_valid.csv', index=False, 
                  columns=['id','user_id','location_id','latitude','longitude','start_day','start_min','weekday'])

## 测试集（pickle格式，与代码中的test_file结构一致）
with open(f'{OUTPUT_DIR}/fsq_testset.pk', 'wb') as f:
    pickle.dump(test_samples, f)

print(f"数据集划分完成：\n"
      f"- 训练集: {len(train_data)}条\n"
      f"- 验证集: {len(valid_data)}条\n"
      f"- 测试样本: {len(test_samples)}组")