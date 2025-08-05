import pandas as pd

# 读取两个CSV文件
main_df = pd.read_csv('d:\\download\\cumcm2020c\\C\\raw\\预处理数据\\附件二企业特征_优化.csv')
predict_df = pd.read_csv('d:\\download\\cumcm2020c\\C\\raw\\预处理数据\\附件二企业信誉评级预测结果.csv')

# 合并数据（根据企业代号）
merged_df = pd.merge(main_df, predict_df, on='企业代号', how='left')

# 调整列顺序，将预测信誉评级放在企业名称后面
columns = merged_df.columns.tolist()
# 找到企业名称的索引位置
name_index = columns.index('企业名称')
# 移除预测信誉评级列
columns.remove('预测信誉评级')
# 在企业名称后面插入预测信誉评级
columns.insert(name_index + 1, '预测信誉评级')
# 重新排序列
merged_df = merged_df[columns]

# 保存合并后的结果
merged_df.to_csv('d:\\download\\cumcm2020c\\C\\raw\\预处理数据\\附件一企业特征_优化_合并后.csv', index=False, encoding='utf-8-sig')
print('数据合并完成，结果已保存到附件一企业特征_优化_合并后.csv')