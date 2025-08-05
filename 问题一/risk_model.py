import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 读取数据
input_path = r'd:\download\cumcm2020c\C\raw\预处理数据\附件一企业特征_优化.csv'
df = pd.read_csv(input_path)

# 数据预处理
# 1. 信誉评级映射为数值 (A:0.1, B:0.3, C:0.5, D:0.7)
rating_mapping = {'A': 0.1, 'B': 0.3, 'C': 0.5, 'D': 0.7}
temp = df['信誉评级']
df['信誉评级'] = df['信誉评级'].map(rating_mapping)
# 2. 选择特征并归一化
features = ['信誉评级', '发票作废率', '负数发票占比', '净现金流', '进销项比', '金额标准差', '月交易额波动率', '最大断月数', '合作企业数量', '活跃天数', '发票总数量', '金额总和', '金额均值', '月交易额平均', '最大月增长', '月增长率中位数', '近3月增长率', '活跃月数']

# 处理净现金流（负值表示高风险，取绝对值后归一化）
# df['净现金流'] = df['净现金流'].abs()

# 归一化到[0,1]
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# 模糊综合评价
# 因素集U = 所选特征
# 评语集V = [低风险, 中风险, 高风险] -> [0.2, 0.5, 0.8]
# 权重集W（等权重）
# 2. 选择特征并归一化
def calculate_ahp_weights():
    # 选取6个关键风险字段
    key_features = ['信誉评级', '发票作废率', '最大断月数', '净现金流', '月交易额波动率', '负数发票占比']
    
    # 6x6判断矩阵（基于风险影响程度评估）
    judgment_matrix = np.array([
        [1, 3, 2, 4, 3, 2],   # 信誉评级
        [1/3, 1, 1/2, 2, 1, 1/2],  # 发票作废率
        [1/2, 2, 1, 3, 2, 1],  # 最大断月数
        [1/4, 1/2, 1/3, 1, 1/2, 1/3],  # 净现金流
        [1/3, 1, 1/2, 2, 1, 1/2],  # 月交易额波动率
        [1/2, 2, 1, 3, 2, 1]   # 负数发票占比
    ])
    
    # 几何平均法计算权重
    n = judgment_matrix.shape[0]
    weights = np.prod(judgment_matrix, axis=1) ** (1/n)
    weights = weights / np.sum(weights)  # 归一化
    
    # 一致性检验
    lambda_max = np.max(np.sum(judgment_matrix * weights, axis=1) / weights)
    ci = (lambda_max - n) / (n - 1)
    cr = ci / 1.24  # 6阶矩阵RI值
    if cr > 0.1:
        print(f"警告：CR={cr:.4f} > 0.1，判断矩阵需调整")
    # 生成权重字典（其他字段权重为0）
    weight_dict = {feature: 0 for feature in features}
    for i, feature in enumerate(key_features):
        weight_dict[feature] = weights[i]
    print(json.dumps(weight_dict, indent=4, ensure_ascii=False))
    return np.array([weight_dict[feature] for feature in features])
    

weights = calculate_ahp_weights()

# 隶属度函数（梯形）
def 隶属度(x, feature):
    if feature not in[ '发票作废率', '负数发票占比', '进销项比',  '月交易额波动率', '最大断月数',]:  # 越高风险越小
        return [max(0, (0.2 - x)/0.2), max(0, min((x-0.1)/0.2, (0.5-x)/0.2)), max(0, (x-0.3)/0.4)]
    else:  # 其他特征越高风险越大
        return [max(0, (0.3 - x)/0.3), max(0, min((x-0.2)/0.3, (0.7-x)/0.3)), max(0, (x-0.5)/0.5)]

# 计算风险系数
risk_coefficients = []
for _, row in df.iterrows():
    # 构建模糊评价矩阵
    eval_matrix = []
    for feature in features:
        eval_matrix.append(隶属度(row[feature], feature))
    eval_matrix = np.array(eval_matrix)
    
    # 模糊合成（加权平均）
    result = np.dot(weights, eval_matrix)
    
    # 去模糊化（重心法）
    risk = np.sum(result * np.array([0.2, 0.5, 0.8])) / np.sum(result)
    risk_coefficients.append(max(0, min(1, risk)))  # 确保在0-1之间

df['风险系数'] = risk_coefficients
df['信誉评级'] = temp
# 保存结果
output_path = r'd:\download\cumcm2020c\C\raw\问题一\企业风险系数结果.csv'
df['评分分数'] =(( 1-df['风险系数'] )* 100).round(2)
df[['企业代号', '企业名称', '风险系数','信誉评级', '评分分数']].to_csv(output_path, index=False,encoding='utf-8-sig')

print(f'风险系数计算完成，结果保存至: {output_path}')