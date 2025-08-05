import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from data_loader import load_attachment1_data, load_attachment2_data
from preprocessor import DataPreprocessor
from model_trainer import RiskModelTrainer
from risk_quantifier import RiskQuantifier

class RiskModelPipeline:
    def __init__(self):
        self.data_preprocessor = DataPreprocessor()
        self.model_trainer = RiskModelTrainer()
        self.risk_quantifier = RiskQuantifier()
        self.attachment1_data = None
        self.attachment2_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self):
        """加载所有数据"""
        print("正在加载数据...")
        self.attachment1_data = load_attachment1_data()
        self.attachment2_data = load_attachment2_data()
        print(f"成功加载数据: 附件1({len(self.attachment1_data)}家企业), 附件2({len(self.attachment2_data)}家企业)")
        return self
        
    def prepare_training_data(self):
        """准备训练数据（基于附件1）"""
        if self.attachment1_data is None:
            self.load_data()
            
        print("正在准备训练数据...")
        # 复制数据
        train_data = self.attachment1_data.copy()
        
        # 从信誉评级创建目标变量：D级为高风险(1)，其他为低风险(0)
        train_data['风险标签'] = train_data['信誉评级'].apply(lambda x: 1 if x == 'D' else 0)
        
        # 提取企业代号
        enterprise_codes = train_data['企业代号']
        
        # 分离特征和目标变量
        X = train_data.drop('风险标签', axis=1)
        y = train_data['风险标签']
        
        # 准备特征
        features = self.data_preprocessor.prepare_features(X)
        
        # 分割训练集和测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            features, y, test_size=0.2, random_state=42
        )
        
        print(f"训练数据准备完成: 训练集{self.X_train.shape}, 测试集{self.X_test.shape}")
        return self
        
    def train_model(self, use_grid_search=False):
        """训练风险模型"""
        if self.X_train is None or self.y_train is None:
            self.prepare_training_data()
            
        print("正在训练风险模型...")
        self.model_trainer.train(self.X_train, self.y_train, use_grid_search)
        
        # 评估模型
        print("\n模型评估结果:")
        self.model_trainer.evaluate(self.X_test, self.y_test)
        
        # 交叉验证
        print("\n交叉验证结果:")
        self.model_trainer.cross_validate(pd.concat([self.X_train, self.X_test]), pd.concat([self.y_train, self.y_test]))
        
        # 保存模型
        self.model_trainer.save_model()
        return self
        
    def quantify_attachment1_risk(self):
        """量化附件1企业的风险"""
        if not hasattr(self.model_trainer.model, 'coef_'):
            self.train_model()
            
        print("\n正在量化附件1企业风险...")
        # 准备所有附件1企业的特征
        all_features = self.data_preprocessor.prepare_features(
            self.attachment1_data, is_training=False
        )
        
        # 预测风险概率
        risk_probabilities = self.model_trainer.model.predict_proba(all_features)[:, 1]
        
        # 量化风险
        risk_results = self.risk_quantifier.quantify_risk(
            risk_probabilities, self.attachment1_data['企业代号']
        )
        
        # 合并原始信誉评级用于对比
        risk_results = risk_results.merge(
            self.attachment1_data[['企业代号', '信誉评级']],
            on='企业代号',
            how='left'
        )
        
        # 保存结果
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '附件1企业风险量化结果.csv')
        risk_results.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"附件1企业风险量化结果已保存到: {output_path}")
        return risk_results
        
    def quantify_attachment2_risk(self):
        """量化附件2企业的风险"""
        if not hasattr(self.model_trainer.model, 'coef_'):
            self.train_model()
            
        print("\n正在量化附件2企业风险...")
        # 准备所有附件2企业的特征
        all_features = self.data_preprocessor.prepare_features(
            self.attachment2_data, is_training=False
        )
        
        # 预测风险概率
        risk_probabilities = self.model_trainer.model.predict_proba(all_features)[:, 1]
        
        # 量化风险
        risk_results = self.risk_quantifier.quantify_risk(
            risk_probabilities, self.attachment2_data['企业代号']
        )
        
        # 保存结果
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '附件2企业风险量化结果.csv')
        risk_results.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"附件2企业风险量化结果已保存到: {output_path}")
        return risk_results
        
    def run_full_pipeline(self, use_grid_search=False):
        """运行完整的风险量化流程"""
        print("====== 开始风险量化模型流程 ======")
        self.load_data()\
            .prepare_training_data()\
            .train_model(use_grid_search)
        
        # 量化附件1和附件2企业风险
        attachment1_results = self.quantify_attachment1_risk()
        attachment2_results = self.quantify_attachment2_risk()
        
        # 生成风险报告
        report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '风险量化报告.txt')
        self.risk_quantifier.generate_risk_report(attachment1_results, report_path)
        
        print("\n====== 风险量化模型流程完成 ======")
        return attachment1_results, attachment2_results

if __name__ == '__main__':
    # 创建并运行风险量化流程
    pipeline = RiskModelPipeline()
    # 设置use_grid_search=True可启用网格搜索优化超参数
    pipeline.run_full_pipeline(use_grid_search=False)