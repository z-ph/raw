import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
# 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False  
# 1. 加载风险系数评分数据
# 假设风险系数存储在"风险量化.csv"的"risk_score"列中
# 根据实际数据路径修改file_path
file_path = os.path.join(os.path.dirname(__file__), "./企业风险系数结果.csv")
risk_data = pd.read_csv(file_path)

# 2. 数据预处理
# 提取风险系数评分列并标准化
X = risk_data[['评分分数']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 执行K-means聚类
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# 4. 将聚类结果添加到原始数据
risk_data['cluster'] = clusters

# 5. 保存聚类结果
output_path = "问题1聚类结果.csv"
risk_data.to_csv(output_path, index=False)
print(f"聚类完成，结果已保存至: {output_path}")

# 6. 可选：可视化聚类结果
# 两个子图的距离拉开
fig, ax = plt.subplots(figsize=(10, 6),ncols=2, sharey=True)
# 将x轴数据从企业编号索引改为信誉评级
ax[0].scatter(risk_data['信誉评级'], X_scaled, c=clusters, cmap='viridis', alpha=0.7)
ax[0].set_xlabel('信誉评级')  # 更新x轴标签
ax[0].set_ylabel('标准化风险系数')
ax[0].set_title('企业风险系数K-means聚类结果(k=4)')
cbar = fig.colorbar(ax[0].scatter(risk_data['信誉评级'], X_scaled, c=clusters, cmap='viridis'), ax=ax[0])
cbar.set_label('聚类标签')
# 新增：以分数为x轴的散点图
ax[1].scatter(risk_data['评分分数'], X_scaled, c=clusters, cmap='viridis', alpha=0.7)
ax[1].set_xlabel('评分分数')
ax[1].set_ylabel('标准化风险系数')
ax[1].set_title('企业风险系数与评分分数关系图(k=4)')
cbar = fig.colorbar(ax[1].scatter(risk_data['评分分数'], X_scaled, c=clusters, cmap='viridis'), ax=ax[1])
cbar.set_label('聚类标签')
plt.savefig('聚类结果.png')
plt.close()