import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置文件路径
episode_path = 'results/run-.-tag-Train_mean_episode_length.csv'  # 替换为第一个CSV文件的路径
reward_path = "results/run-.-tag-Train_mean_reward.csv"

# 读取数据
df_episode = pd.read_csv(episode_path)
df_reward = pd.read_csv(reward_path)

# 过滤数据，只保留Step <= 60000的行
max_step = 59000
df_episode = df_episode[df_episode['Step'] <= max_step]
df_reward = df_reward[df_reward['Step'] <= max_step]

# 创建图形和子图
plt.figure(figsize=(14, 6))  # 更宽的图形以适应两个子图

# 第一个子图：Episode Length
plt.subplot(1, 2, 1)  # 1行2列的第1个子图
sns.lineplot(data=df_episode, x='Step', y='Value', color='royalblue', linewidth=2)
plt.title('Mean Episode Length', fontsize=14)
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Length', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.3)
sns.despine()

# 第二个子图：Reward
plt.subplot(1, 2, 2)  # 1行2列的第2个子图
sns.lineplot(data=df_reward, x='Step', y='Value', color='coral', linewidth=2)
plt.title('Mean Reward', fontsize=14)
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Reward', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.3)
sns.despine()

# 调整布局
plt.tight_layout(pad=3.0)  # 增加子图之间的间距

# 保存为SVG格式
plt.savefig('rl_training_metrics.svg', format='svg', dpi=1200, bbox_inches='tight')

# 显示图形
plt.show()