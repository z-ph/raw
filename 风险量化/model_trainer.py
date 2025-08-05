import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV

class RiskModelTrainer:
    def __init__(self):
        self.model = LogisticRegression(
            penalty='l2',  # L2正则化
            C=1.0,          # 正则化强度的倒数
            solver='liblinear',  # 适合小数据集的求解器
            max_iter=1000,  # 增加迭代次数确保收敛
            class_weight='balanced',  # 处理不平衡数据
            random_state=42
        )
        self.best_params_ = None
        self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'risk_model.pkl')
        
    def train(self, X_train, y_train, use_grid_search=False):
        """训练模型"""
        if use_grid_search:
            # 网格搜索优化超参数
            param_grid = {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
            grid_search = GridSearchCV(
                estimator=self.model,
                param_grid=param_grid,
                cv=5,
                scoring='roc_auc',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            self.best_params_ = grid_search.best_params_
            self.model = grid_search.best_estimator_
            print(f"最佳超参数: {self.best_params_}")
        else:
            # 使用默认参数训练
            self.model.fit(X_train, y_train)
        return self.model
        
    def evaluate(self, X_test, y_test):
        """评估模型性能"""
        if not hasattr(self.model, 'coef_'):
            raise ValueError("模型尚未训练，请先调用train方法")
            
        # 预测概率和类别
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = self.model.predict(X_test)
        
        # 计算评估指标
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        
        # 打印评估结果
        print(f"准确率: {accuracy:.4f}")
        print(f"AUC: {auc:.4f}")
        print("混淆矩阵:")
        print(conf_matrix)
        print("分类报告:")
        print(class_report)
        
        # 返回评估指标
        return {
            'accuracy': accuracy,
            'auc': auc,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report
        }
        
    def cross_validate(self, X, y, cv=5):
        """交叉验证评估模型"""
        scores = cross_val_score(
            self.model, X, y, cv=cv, scoring='roc_auc'
        )
        print(f"交叉验证AUC: {scores.mean():.4f} ± {scores.std():.4f}")
        return scores
        
    def save_model(self, file_path=None):
        """保存模型到文件"""
        if not hasattr(self.model, 'coef_'):
            raise ValueError("模型尚未训练，请先调用train方法")
            
        save_path = file_path if file_path else self.model_path
        joblib.dump(self.model, save_path)
        print(f"模型已保存到: {save_path}")
        return save_path
        
    def load_model(self, file_path=None):
        """从文件加载模型"""
        load_path = file_path if file_path else self.model_path
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"模型文件不存在: {load_path}")
            
        self.model = joblib.load(load_path)
        print(f"已从{load_path}加载模型")
        return self.model
        
    def get_feature_importance(self, feature_names):
        """获取特征重要性"""
        if not hasattr(self.model, 'coef_'):
            raise ValueError("模型尚未训练，请先调用train方法")
            
        # 逻辑回归的系数即为特征重要性
        importance = pd.DataFrame({
            '特征名称': feature_names,
            '系数': self.model.coef_[0]
        })
        importance = importance.sort_values('系数', ascending=False)
        return importance