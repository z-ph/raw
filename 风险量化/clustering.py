import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

# 读取预处理数据
data = pd.read_csv('d:/download/cumcm2020c/C/raw/预处理数据/附件一企业特征_优化.csv')

# 选择数值特征列进行聚类
features = data.select_dtypes(include=['float64', 'int64'])

# 处理缺失值
imputer = SimpleImputer(strategy='mean')
features_imputed = imputer.fit_transform(features)

# 应用k-means聚类，设置k=4，随机数种子为42
# 注意：聚类中心不是唯一确定的，不同的k或随机数种子可能会产生不同的聚类结果
kmeans = KMeans(n_clusters=4, random_state=43)
data['cluster'] = kmeans.fit_predict(features_imputed)

# 打印企业ID和对应的聚类结果
# print("企业代号,聚类结果")
# for _, row in data.iterrows():
#     print(f"{row['企业代号']},{row['cluster']}")

# 计算并打印每个聚类中的评级占比
print("\n每个聚类中各个评级的数量:")
rating_counts = data.groupby(['cluster', '信誉评级']).size().unstack(fill_value=0)
print(rating_counts)