# 中小微企业信贷风险量化模型

## 项目概述
本项目针对银行中小微企业信贷决策问题，构建了一套风险量化模型和信贷策略优化系统。模型基于企业交易数据和信誉评级，实现了风险评估与贷款额度的智能分配，可有效支持银行在信贷总额约束下的科学化决策。

## 环境要求
- Python 3.8+
- 依赖库：
  ```
  pandas==2.3.1
  numpy==2.2.2
  scikit-learn==1.7.1
  matplotlib==3.9.0
  ```
- 安装方法：`pip install -r requirements.txt`

## 文件结构
```
.\
├── 风险量化.py          # 主程序
├── 题目.md             # 问题描述
├── 思路.md             # 建模思路文档
├── 附件1-3/...         # 原始数据文件
├── 问题1信贷策略.csv    # 问题1结果输出
├── 问题2信贷策略.csv    # 问题2结果输出
└── README.md           # 说明文档
```

## 模型架构
### 1. 数据处理模块
**功能**：加载并预处理企业数据，提取风险特征
**核心代码**：
```python
def preprocess_data(enterprise_info, sales_invoices, purchase_invoices):
    # 计算销售总额、采购总额
    sales_summary = sales_invoices.groupby('企业代号')['金额'].sum().reset_index()
    purchase_summary = purchase_invoices.groupby('企业代号')['金额'].sum().reset_index()
    
    # 计算有效发票占比
    valid_invoice_ratio = sales_invoices.groupby('企业代号').apply(
        lambda x: sum(x['发票状态'] == '有效发票') / len(x)
    ).reset_index()
    
    # 特征合并
    enterprise_features = enterprise_info.merge(sales_summary, on='企业代号', how='left')
    enterprise_features = enterprise_features.merge(purchase_summary, on='企业代号', how='left')
    enterprise_features = enterprise_features.merge(valid_invoice_ratio, on='企业代号', how='left')
    
    return enterprise_features
```

### 2. 风险量化模型
**核心思想**：基于企业财务特征和信誉评级，构建逻辑回归模型预测违约风险
**实现流程**：
1. **特征工程**：
   - 财务指标：销售总额、采购总额、销售采购比
   - 交易质量：有效发票占比
   - 信誉评级：A/B/C映射为3/2/1数值
   - 行业风险：个体经营+0.3，公司+0.1（针对无信贷记录企业）

2. **模型训练**：
```python
def build_risk_model(enterprise_features):
    # 创建风险特征
    enterprise_features['销售采购比'] = enterprise_features['销售总额'] / (enterprise_features['采购总额'] + 1e-6)
    
    # 信誉评级映射
    rating_map = {'A': 3, 'B': 2, 'C': 1, 'D': 0}
    enterprise_features['信誉评级数值'] = enterprise_features['信誉评级'].map(rating_map)
    
    # 训练逻辑回归模型
    model = LogisticRegression()
    model.fit(X_scaled, y)  # X为特征矩阵，y为是否违约标签
    
    # 输出风险分数（违约概率）
    enterprise_features['风险分数'] = model.predict_proba(X_scaled)[:, 1]
```

### 3. 信贷决策模型
**优化目标**：在信贷总额约束下，最大化低风险企业的贷款覆盖率
**决策规则**：
1. 按风险分数升序排序（低风险优先）
2. 额度分配标准：
   - 风险分数 < 0.2 → 100万元
   - 0.2 ≤ 风险分数 < 0.5 → 50万元
   - 风险分数 ≥ 0.5 → 10万元
3. 累计额度不超过设定阈值（如1亿元）

**代码实现**：
```python
def optimize_credit_strategy(enterprise_features, total_credit=1e8):
    # 按风险排序
    enterprise_features = enterprise_features.sort_values('风险分数')
    
    # 额度分配
    enterprise_features['贷款额度'] = np.where(
        enterprise_features['风险分数'] < 0.2, 1000000,
        np.where(enterprise_features['风险分数'] < 0.5, 500000, 100000)
    )
    
    # 总额控制
    enterprise_features['累计额度'] = enterprise_features['贷款额度'].cumsum()
    eligible_enterprises = enterprise_features[enterprise_features['累计额度'] <= total_credit]
    return eligible_enterprises
```

## 使用指南
### 基本流程
1. 准备数据：确保附件1-3数据文件在项目目录下
2. 安装依赖：`pip install -r requirements.txt`
3. 运行模型：`python 风险量化.py`
4. 查看结果：生成的CSV文件包含推荐贷款企业及额度

### 参数说明
- `total_credit`：总信贷额度（默认1亿元）
- `risk_threshold`：风险分数阈值（默认0.2/0.5）

## 结果说明
### 输出文件
- **问题1信贷策略.csv**：123家有信贷记录企业的贷款建议
- **问题2信贷策略.csv**：302家无信贷记录企业的贷款建议

### 结果字段
| 字段       | 说明                 |
|------------|----------------------|
| 企业代号   | 企业唯一标识         |
| 企业名称   | 企业全称             |
| 信誉评级   | 银行内部信誉评级     |
| 风险分数   | 模型预测的违约概率   |
| 贷款额度   | 建议贷款金额（元）   |
| 累计额度   | 累计贷款总额         |

## 模型改进方向
1. **利率优化**：整合附件3数据，建立利率-风险-流失率的联动模型
2. **行业细分**：细化行业分类，建立行业专属风险评估标准
3. **时间因素**：引入发票时间序列特征，分析企业经营趋势
4. **模型融合**：尝试随机森林、XGBoost等模型提升预测精度

## 注意事项
- 附件2部分文件读取可能失败，代码已做异常处理
- 无信贷记录企业的信誉评级采用默认值，可根据实际情况调整
- 风险分数阈值可根据银行风险偏好灵活调整