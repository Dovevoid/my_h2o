import joblib
import numpy as np
import random

def split_and_combine_dict_data(stable_pkl_path, amass_pkl_path, output_pkl_path):
    # 加载两个PKL文件
    with open(stable_pkl_path, 'rb') as f:
        stable_data = joblib.load(f)
    
    with open(amass_pkl_path, 'rb') as f:
        amass_data = joblib.load(f)
    
    # 检查数据是否为字典格式
    if not (isinstance(stable_data, dict) and isinstance(amass_data, dict)):
        raise ValueError("输入文件应该包含字典格式的数据")
    
    # 获取stable数据的所有动作名
    stable_keys = list(stable_data.keys())
    print(len(stable_keys))
    num_stable_actions = len(stable_keys)
    half_stable = num_stable_actions // 2
    
    # 分割stable数据的前一半
    first_half_keys = stable_keys[:half_stable]
    first_half = {k: stable_data[k] for k in first_half_keys}
    
    # 从amass数据中随机选取与另一半数量相同的动作
    amass_keys = list(amass_data.keys())
    print(amass_keys[:6])
    num_needed = num_stable_actions - half_stable
    
    if len(amass_keys) > num_needed:
        # 如果amass数据足够，随机选择不重复的键
        selected_keys = random.sample(amass_keys, num_needed)
    else:
        # 如果amass数据不足，全部使用（可能会重复）
        selected_keys = amass_keys * (num_needed // len(amass_keys)) + amass_keys[:num_needed % len(amass_keys)]
    
    second_half = {k: amass_data[k] for k in selected_keys}
    
    # 合并两部分数据
    combined_data = {**first_half, **second_half}
    
    # 保存结果到新的PKL文件
    # with open(output_pkl_path, 'wb') as f:
    #     joblib.dump(combined_data, f)
    
    print(f"操作完成! 新文件包含 {len(combined_data)} 个动作数据:")
    print(f"- 来自 {stable_pkl_path}: {len(first_half)} 个")
    print(f"- 来自 {amass_pkl_path}: {len(second_half)} 个")
    print(f"新文件已保存为: {output_pkl_path}")

# 使用示例
stable_pkl_path = '/home/peter/h2o/human2humanoid-main/legged_gym/resources/motions/h1/stable_punch.pkl'
amass_pkl_path = 'human2humanoid-main/legged_gym/resources/motions/h1/amass_phc_filtered.pkl'
output_pkl_path = 'combined_actions_dict.pkl'

split_and_combine_dict_data(stable_pkl_path, amass_pkl_path, output_pkl_path)