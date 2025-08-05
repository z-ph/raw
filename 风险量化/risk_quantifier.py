import pandas as pd
import numpy as np

class RiskQuantifier:
    def __init__(self, risk_thresholds=None):
        """
        初始化风险量化器
        
        参数:
        risk_thresholds: dict - 风险等级阈值，默认为{'低风险': 0.3, '中风险': 0.6, '高风险': 1.0}
        """
        # 默认风险等级阈值
        self.risk_thresholds = risk_thresholds or {
            '低风险': 0.3,
            '中风险': 0.6,
            '高风险': 1.0
        }
        # 确保阈值按升序排列
        self.sorted_thresholds = sorted(self.risk_thresholds.items(), key=lambda x: x[1])
        
    def calculate_risk_score(self, probability):
        """
        将风险概率转换为0-100的风险分数
        
        参数:
        probability: float - 模型输出的违约概率(0-1)
        
        返回:
        int - 风险分数(0-100)，分数越高风险越大
        """
        if not 0 <= probability <= 1:
            raise ValueError("概率值必须在0到1之间")
        # 将概率线性映射到0-100分
        return int(round(probability * 100))
        
    def determine_risk_level(self, probability):
        """
        根据风险概率确定风险等级
        
        参数:
        probability: float - 模型输出的违约概率(0-1)
        
        返回:
        str - 风险等级名称
        """
        if not 0 <= probability <= 1:
            raise ValueError("概率值必须在0到1之间")
            
        for level, threshold in self.sorted_thresholds:
            if probability <= threshold:
                return level
        return self.sorted_thresholds[-1][0]
        
    def quantify_risk(self, probabilities, enterprise_codes):
        """
        批量量化风险，生成包含风险分数和等级的DataFrame
        
        参数:
        probabilities: array-like - 模型输出的违约概率数组
        enterprise_codes: array-like - 对应的企业代号数组
        
        返回:
        pd.DataFrame - 包含企业代号、风险概率、风险分数和风险等级的DataFrame
        """
        if len(probabilities) != len(enterprise_codes):
            raise ValueError("概率数组和企业代号数组长度必须一致")
            
        # 创建结果DataFrame
        risk_results = pd.DataFrame({
            '企业代号': enterprise_codes,
            '风险概率': probabilities,
            '风险分数': [self.calculate_risk_score(p) for p in probabilities],
            '风险等级': [self.determine_risk_level(p) for p in probabilities]
        })
        
        # 按风险分数降序排序
        risk_results = risk_results.sort_values('风险分数', ascending=False)
        return risk_results
        
    def adjust_for_industry(self, risk_results, industry_risk_factors):
        """
        根据行业风险因素调整风险评估结果
        
        参数:
        risk_results: pd.DataFrame - 原始风险评估结果
        industry_risk_factors: dict - 行业风险调整因子，如{'制造业': 1.1, '服务业': 0.9}
        
        返回:
        pd.DataFrame - 调整后的风险评估结果
        """
        if '行业' not in risk_results.columns:
            raise ValueError("风险结果DataFrame必须包含'行业'列")
            
        risk_results_copy = risk_results.copy()
        
        # 应用行业调整因子
        risk_results_copy['调整后风险分数'] = risk_results_copy.apply(
            lambda row: min(100, int(round(row['风险分数'] * industry_risk_factors.get(row['行业'], 1.0)))),
            axis=1
        )
        
        # 根据调整后的风险分数重新确定风险等级
        risk_results_copy['调整后风险等级'] = risk_results_copy['调整后风险分数'].apply(
            lambda score: self.determine_risk_level(score / 100)
        )
        
        return risk_results_copy
        
    def generate_risk_report(self, risk_results, output_path=None):
        """
        生成风险评估报告
        
        参数:
        risk_results: pd.DataFrame - 风险评估结果
        output_path: str - 报告输出路径，如为None则不保存文件
        
        返回:
        dict - 风险统计摘要
        """
        # 计算风险等级分布
        risk_distribution = risk_results['风险等级'].value_counts().to_dict()
        
        # 计算平均风险分数
        avg_risk_score = risk_results['风险分数'].mean()
        
        # 找出最高风险的企业
        top_risk_enterprises = risk_results.head(5)['企业代号'].tolist()
        
        # 生成报告摘要
        report_summary = {
            '风险等级分布': risk_distribution,
            '平均风险分数': round(avg_risk_score, 2),
            '高风险企业数量': risk_distribution.get('高风险', 0),
            '最高风险企业前5名': top_risk_enterprises
        }
        
        # 如果提供了输出路径，则保存报告
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("# 企业信贷风险评估报告\n\n")
                f.write(f"## 总体风险概况\n")
                f.write(f"- 平均风险分数: {report_summary['平均风险分数']}\n")
                f.write(f"- 高风险企业数量: {report_summary['高风险企业数量']}\n\n")
                
                f.write("## 风险等级分布\n")
                for level, count in report_summary['风险等级分布'].items():
                    f.write(f"- {level}: {count}家企业\n")
                
                f.write("\n## 最高风险企业\n")
                for i, enterprise in enumerate(top_risk_enterprises, 1):
                    f.write(f"{i}. {enterprise}\n")
            print(f"风险报告已保存到: {output_path}")
        
        return report_summary