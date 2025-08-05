import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import minimize_scalar
from sklearn.impute import SimpleImputer
base_dir = 'D:/download/cumcm2020c/C/raw/源数据'
fileList = {
    '附件一':[base_dir+'/附件1：123家有信贷记录企业的相关数据.xlsx_0.csv',base_dir+'/附件1：123家有信贷记录企业的相关数据.xlsx_1.csv',base_dir+'/附件1：123家有信贷记录企业的相关数据.xlsx_2.csv'],
    '附件二':[base_dir+'/附件2：302家无信贷记录企业的相关数据.xlsx_0.csv',base_dir+'/附件2：302家无信贷记录企业的相关数据.xlsx_1.csv',base_dir+'/附件2：302家无信贷记录企业的相关数据.xlsx_2.csv'],

    '附件三':[base_dir+'/附件3：银行贷款年利率与客户流失率关系的统计数据.xlsx_0.csv'],
}
# 数据加载模块
def load_data():
    # 加载附件1数据
    enterprise_info = pd.read_csv(fileList['附件一'][0],encoding='GBK')
    sales_invoices = pd.read_csv(fileList['附件一'][1],encoding='GBK')
    purchase_invoices = pd.read_csv(fileList['附件一'][2],encoding='GBK')
    
    # 加载附件2数据（无信贷记录企业）
    try:
        non_credit_enterprises = pd.read_csv(fileList['附件二'][0],encoding='GBK')
        # 尝试加载附件2的发票数据（处理可能的读取错误）
        try:
            non_credit_sales = pd.read_csv(fileList['附件二'][1],encoding='GBK')
        except:
            non_credit_sales = pd.DataFrame(columns=['企业代号', '金额', '发票状态'])
        
        try:
            non_credit_purchases = pd.read_csv(fileList['附件二'][2],encoding='GBK')
        except:
            non_credit_purchases = pd.DataFrame(columns=['企业代号', '金额', '发票状态'])
    except Exception as e:
        print(f"附件2数据加载失败: {e}")
        non_credit_enterprises = pd.DataFrame(columns=['企业代号', '企业名称'])
        non_credit_sales = pd.DataFrame()
        non_credit_purchases = pd.DataFrame()
    
    # 加载附件3数据
    interest_rates = pd.read_csv(fileList['附件三'][0],encoding='GBK')
    
    return enterprise_info, sales_invoices, purchase_invoices, interest_rates, non_credit_enterprises, non_credit_sales, non_credit_purchases

# 数据预处理模块
def preprocess_data(enterprise_info, sales_invoices, purchase_invoices):
    # 计算企业销售总额
    sales_summary = sales_invoices.groupby('企业代号')['金额'].sum().reset_index()
    sales_summary.columns = ['企业代号', '销售总额']
    
    # 计算企业采购总额
    purchase_summary = purchase_invoices.groupby('企业代号')['金额'].sum().reset_index()
    purchase_summary.columns = ['企业代号', '采购总额']
    
    # 计算有效发票占比
    valid_invoice_ratio = sales_invoices.groupby('企业代号').apply(
        lambda x: sum(x['发票状态'] == '有效发票') / len(x)
    ).reset_index()
    valid_invoice_ratio.columns = ['企业代号', '有效发票占比']
    
    # 合并企业特征
    enterprise_features = enterprise_info.merge(sales_summary, on='企业代号', how='left')
    enterprise_features = enterprise_features.merge(purchase_summary, on='企业代号', how='left')
    enterprise_features = enterprise_features.merge(valid_invoice_ratio, on='企业代号', how='left')
    
    # 处理缺失值
    enterprise_features.fillna(0, inplace=True)
    
    return enterprise_features

# 风险量化模型
def build_risk_model(enterprise_features):
    # 创建风险特征
    enterprise_features['销售采购比'] = enterprise_features['销售总额'] / (enterprise_features['采购总额'] + 1e-6)
    
    # 信誉评级映射
    rating_map = {'A': 3, 'B': 2, 'C': 1, 'D': 0}
    enterprise_features['信誉评级数值'] = enterprise_features['信誉评级'].map(rating_map)
    
    # 违约状态映射
    enterprise_features['是否违约数值'] = enterprise_features['是否违约'].map({'是': 1, '否': 0})
    
    # 选择特征
    features = ['信誉评级数值', '销售总额', '采购总额', '有效发票占比', '销售采购比']
    X = enterprise_features[features]
    y = enterprise_features['是否违约数值']
    
    # 特征标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 训练逻辑回归模型
    model = LogisticRegression()
    model.fit(X_scaled, y)
    
    # 计算风险分数
    enterprise_features['风险分数'] = model.predict_proba(X_scaled)[:, 1]
    
    return enterprise_features, model, scaler

# 信贷策略优化
def optimize_credit_strategy(enterprise_features, total_credit=1e8):
    # 根据风险分数排序，风险越低优先级越高
    enterprise_features = enterprise_features.sort_values('风险分数')
    
    # 分配贷款额度（10-100万）
    enterprise_features['贷款额度'] = np.where(
        enterprise_features['风险分数'] < 0.2, 1000000,  # 低风险企业：100万
        np.where(enterprise_features['风险分数'] < 0.5, 500000,  # 中风险企业：50万
                 100000)  # 高风险企业：10万
    )
    
    # 计算累计贷款额度
    enterprise_features['累计额度'] = enterprise_features['贷款额度'].cumsum()
    
    # 根据总信贷额度限制筛选企业
    eligible_enterprises = enterprise_features[enterprise_features['累计额度'] <= total_credit]
    
    return eligible_enterprises

# 利率决策模型 - 数据预处理
def preprocess_interest_rate_data(interest_rates):
    # 处理表头，提取信誉评级列
    rate_df = interest_rates.iloc[1:].copy()
    rate_df.columns = ['贷款年利率', '信誉评级A流失率', '信誉评级B流失率', '信誉评级C流失率']
    
    # 转换为数值类型
    for col in rate_df.columns:
        rate_df[col] = pd.to_numeric(rate_df[col], errors='coerce')
    
    # 填充缺失值
    rate_df = rate_df.dropna()
    
    return rate_df

# 利率决策模型 - 拟合流失率曲线
def fit_churn_rate_models(rate_df):
    models = {}
    # 对每个信誉评级分别拟合多项式模型
    for rating in ['A', 'B', 'C']:
        # 提取数据
        X = rate_df['贷款年利率'].values.reshape(-1, 1)
        y = rate_df[f'信誉评级{rating}流失率'].values
        
        # 创建多项式特征（二次项）
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        
        # 拟合模型
        model = LinearRegression()
        model.fit(X_poly, y)
        
        # 保存模型和多项式转换器
        models[rating] = {"model": model, "poly": poly}
    
    return models

# 利率决策模型 - 计算最优利率
def calculate_optimal_interest_rate(enterprise, models, risk_score, loan_amount):
    # 获取信誉评级（默认A）
    rating = enterprise.get('信誉评级', 'A')
    if rating not in models:
        rating = 'B'  # 默认为B类
    
    # 获取模型
    model_data = models[rating]
    model = model_data['model']
    poly = model_data['poly']
    
    # 定义目标函数：最大化收益
    def objective_function(rate):
        # 确保利率在4%-15%范围内
        if rate < 0.04 or rate > 0.15:
            return -np.inf  # 超出范围返回负无穷
        
        # 预测流失率
        X_poly = poly.transform(np.array([[rate]]))
        churn_rate = model.predict(X_poly)[0]
        
        # 违约概率（风险分数）
        default_prob = risk_score
        
        # 银行收益计算
        # 收益 = (贷款金额 * 利率) * (1 - 流失率) * (1 - 违约概率) - 贷款金额 * 违约概率 * (1 - 流失率)
        revenue = (loan_amount * rate) * (1 - churn_rate) * (1 - default_prob) - loan_amount * default_prob * (1 - churn_rate)
        
        return -revenue  # 最小化负收益（即最大化收益）
    
    # 优化利率
    result = minimize_scalar(objective_function, bounds=(0.04, 0.15), method='bounded')
    
    return round(result.x, 4)  # 保留四位小数

# 主函数
def main():
    # 加载数据
    enterprise_info, sales_invoices, purchase_invoices, interest_rates, non_credit_enterprises, non_credit_sales, non_credit_purchases = load_data()
    
    # 预处理有信贷记录企业数据
    enterprise_features = preprocess_data(enterprise_info, sales_invoices, purchase_invoices)
    
    # 构建风险模型
    enterprise_features, model, scaler = build_risk_model(enterprise_features)
    
    # 优化信贷策略 - 问题1
    credit_strategy = optimize_credit_strategy(enterprise_features)
    
    # 利率决策模型 - 处理利率数据并拟合模型
    rate_df = preprocess_interest_rate_data(interest_rates)
    if not rate_df.empty:
        churn_models = fit_churn_rate_models(rate_df)
        
        # 为每个企业计算最优利率
        credit_strategy['最优利率'] = credit_strategy.apply(
            lambda row: calculate_optimal_interest_rate(
                {'信誉评级': row['信誉评级']}, 
                churn_models, 
                row['风险分数'], 
                row['贷款额度']
            ), axis=1
        )
        
        # 保存包含利率的结果
        credit_strategy.to_csv('问题1信贷利率策略.csv', index=False)
        print('问题1信贷利率策略已保存')
    
    # 处理无信贷记录企业 - 问题2
    if not non_credit_enterprises.empty:
        # 预处理无信贷记录企业数据
        non_credit_features = preprocess_data(non_credit_enterprises, non_credit_sales, non_credit_purchases)
        
        # 为无信贷记录企业添加行业风险因子
        non_credit_features['行业'] = non_credit_features['企业名称'].apply(lambda x: '个体经营' if '个体经营' in x else '公司')
        industry_risk = {'个体经营': 0.3, '公司': 0.1}
        non_credit_features['行业风险'] = non_credit_features['行业'].map(industry_risk)
        
        # 使用已有模型预测风险
        if '信誉评级数值' not in non_credit_features.columns:
            non_credit_features['信誉评级数值'] = 2  # 假设平均信誉评级
            non_credit_features['销售总额'] = non_credit_features.get('销售总额', 0)
            non_credit_features['采购总额'] = non_credit_features.get('采购总额', 0)
            non_credit_features['有效发票占比'] = non_credit_features.get('有效发票占比', 0.8)
            non_credit_features['销售采购比'] = non_credit_features['销售总额'] / (non_credit_features['采购总额'] + 1e-6)
            
        # 处理缺失值
        imputer = SimpleImputer(strategy='mean')
        features = ['信誉评级数值', '销售总额', '采购总额', '有效发票占比', '销售采购比']
        non_credit_features[features] = imputer.fit_transform(non_credit_features[features])
        
        # 预测风险分数
        non_credit_features['风险分数'] = model.predict_proba(scaler.transform(non_credit_features[features]))[:, 1]
        
        # 考虑行业风险
        non_credit_features['风险分数'] = non_credit_features['风险分数'] + non_credit_features['行业风险']
        
        # 问题2：1亿元信贷额度分配
        problem2_strategy = optimize_credit_strategy(non_credit_features, total_credit=100000000)
        
        # 为无信贷记录企业计算最优利率
        if not rate_df.empty and 'churn_models' in locals():
            problem2_strategy['最优利率'] = problem2_strategy.apply(
                lambda row: calculate_optimal_interest_rate(
                    {'信誉评级': 'B'},  # 无信贷记录默认为B类
                    churn_models, 
                    row['风险分数'], 
                    row['贷款额度']
                ), axis=1
            )
        
        problem2_strategy.to_csv('问题2信贷策略.csv', index=False)
        print('问题2信贷策略已保存')
    
    # 输出结果
    print('信贷策略优化结果：')
    print(credit_strategy[['企业代号', '企业名称', '信誉评级', '风险分数', '贷款额度', '最优利率']].head())
    
    # 保存结果
    credit_strategy.to_csv('问题1信贷策略.csv', index=False)
    print('问题1信贷策略结果已保存至CSV文件')

if __name__ == '__main__':
    main()