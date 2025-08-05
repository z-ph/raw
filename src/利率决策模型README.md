# 利率决策模型说明文档

## 1. 模型概述
利率决策模型是风险量化系统的重要组成部分，基于银行贷款年利率与客户流失率的关系数据，为不同风险等级的企业计算最优贷款利率，以最大化银行收益。

## 2. 核心算法详解
### 2.1 逻辑回归（Logistic Regression）
#### 用途
企业违约风险预测，计算企业违约概率（风险分数）

#### 实现位置
<mcsymbol name="build_risk_model" filename="风险量化.py" path="d:\download\cumcm2020c\C\raw\风险量化.py" startline="67" type="function"></mcsymbol>

#### 输入
- 基础特征：信誉评级（A/B/C/D）、是否违约（是/否）
- 财务特征：销售总额、采购总额、有效发票占比、销售采购比

#### 输出
- 风险分数：企业违约概率（0-1之间）

#### 核心代码
```python
# 训练逻辑回归模型
model = LogisticRegression()
model.fit(X_scaled, y)

# 计算风险分数
enterprise_features['风险分数'] = model.predict_proba(X_scaled)[:, 1]
```

### 2.2 多项式回归（Polynomial Regression）
#### 用途
拟合不同信誉评级企业的利率-客户流失率关系曲线

#### 实现位置
<mcsymbol name="fit_churn_rate_models" filename="风险量化.py" path="d:\download\cumcm2020c\C\raw\风险量化.py" startline="147" type="function"></mcsymbol>

#### 输入
- 贷款年利率（4%-15%）
- 不同信誉评级（A/B/C）对应的客户流失率

#### 输出
- 二次多项式模型：流失率 = α×利率² + β×利率 + γ

#### 核心代码
```python
# 创建多项式特征（二次项）
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# 拟合模型
model = LinearRegression()
model.fit(X_poly, y)
```

### 2.3 数值优化算法
#### 用途
计算最优贷款利率，最大化银行收益

#### 实现位置
<mcsymbol name="calculate_optimal_interest_rate" filename="风险量化.py" path="d:\download\cumcm2020c\C\raw\风险量化.py" startline="176" type="function"></mcsymbol>

#### 输入
- 企业信誉评级
- 风险分数（来自逻辑回归模型）
- 贷款额度（来自信贷策略优化）
- 流失率模型（来自多项式回归）

#### 输出
- 最优贷款利率（4%-15%区间内）

#### 核心代码
```python
# 优化利率
result = minimize_scalar(objective_function, bounds=(0.04, 0.15), method='bounded')
return round(result.x, 4)
```

### 2.4 启发式规则算法
#### 用途
信贷额度分配，在总额约束下最大化风险调整后收益

#### 实现位置
<mcsymbol name="optimize_credit_strategy" filename="风险量化.py" path="d:\download\cumcm2020c\C\raw\风险量化.py" startline="98" type="function"></mcsymbol>

#### 输入
- 企业风险分数
- 总信贷额度约束

#### 输出
- 各企业贷款额度
- 累计额度控制

#### 核心代码
```python
# 分配贷款额度（10-100万）
enterprise_features['贷款额度'] = np.where(
    enterprise_features['风险分数'] < 0.2, 1000000,  # 低风险企业：100万
    np.where(enterprise_features['风险分数'] < 0.5, 500000,  # 中风险企业：50万
             100000)  # 高风险企业：10万
)
```

## 3. 实现步骤
### 3.1 数据预处理
```python
rate_df = preprocess_interest_rate_data(interest_rates)
```
- 清洗并转换利率-流失率数据
- 提取信誉评级A/B/C对应的流失率指标
- 处理缺失值和异常值

### 3.2 模型训练
```python
churn_models = fit_churn_rate_models(rate_df)
```
- 为A/B/C类企业分别训练二次多项式回归模型
- 保存模型参数和多项式转换器

### 3.3 最优利率计算
```python
optimal_rate = calculate_optimal_interest_rate(enterprise, churn_models, risk_score, loan_amount)
```
- 输入企业信誉评级、风险分数和贷款额度
- 基于企业信誉评级选择对应模型
- 优化计算最优利率

## 4. 文件说明
- **风险量化.py**：包含利率决策模型的完整实现代码
  - preprocess_interest_rate_data：数据预处理函数
  - fit_churn_rate_models：流失率模型拟合函数
  - calculate_optimal_interest_rate：最优利率计算函数
- **问题1信贷利率策略.csv**：有信贷记录企业的最优利率计算结果
- **问题2信贷策略.csv**：无信贷记录企业的最优利率计算结果

## 5. 使用方法
### 5.1 环境依赖
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- scipy >= 1.7.0

### 5.2 运行步骤
1. 准备好所有输入数据文件
2. 运行风险量化.py脚本：
```bash
python 风险量化.py
```
3. 查看输出的CSV文件获取最优利率结果

## 6. 结果说明
输出CSV文件包含以下关键字段：
- **企业代号**：企业唯一标识
- **信誉评级**：企业信用等级(A/B/C)
- **风险分数**：企业违约概率预测值
- **贷款额度**：分配给企业的贷款金额
- **最优利率**：模型计算的最优贷款利率(百分比)

## 7. 注意事项
- 模型假设利率与流失率呈二次函数关系，实际应用中可根据数据调整多项式阶数
- 最优利率约束在4%-15%之间，如需调整可修改calculate_optimal_interest_rate函数中的bounds参数
- 无信贷记录企业默认按B类信誉评级计算，如有实际数据可调整评级映射规则
- 运行时可能出现FutureWarning警告，不影响结果正确性