import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoder = OrdinalEncoder(categories=[['A', 'B', 'C', 'D']])
        self.imputer = SimpleImputer(strategy='median')
        self.feature_columns = None
        
    def handle_missing_values(self, df):
        """处理缺失值"""
        # 对数值型特征使用中位数填充
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_cols] = self.imputer.fit_transform(df[numeric_cols])
        return df
        
    def encode_categorical_features(self, df):
        """编码分类特征"""
        # 处理信誉评级
        if '信誉评级' in df.columns:
            df['信誉评级_encoded'] = self.encoder.fit_transform(df[['信誉评级']])
            df = df.drop('信誉评级', axis=1)
        else:
            # 对于无信誉评级的数据，添加默认编码值
            df['信誉评级_encoded'] = 0
        return df
        
    def scale_numerical_features(self, df):
        """缩放数值特征"""
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        return df
        
    def drop_unnecessary_columns(self, df):
        """删除不需要的列"""
        # 企业代号作为标识符，不参与建模
        if '企业代号' in df.columns:
            self.feature_columns = df.columns.drop('企业代号')
            return df.drop('企业代号', axis=1)
        self.feature_columns = df.columns
        return df
        
    def prepare_features(self, df, is_training=True):
        """完整的特征准备流程"""
        # 复制数据以避免修改原始数据
        df_copy = df.copy()
        
        # 删除不需要的列
        df_copy = self.drop_unnecessary_columns(df_copy)
        
        # 处理缺失值
        df_copy = self.handle_missing_values(df_copy)
        
        # 编码分类特征
        df_copy = self.encode_categorical_features(df_copy)
        
        # 缩放数值特征
        if is_training:
            df_copy = self.scale_numerical_features(df_copy)
        else:
            # 对于测试集，使用训练时拟合的scaler
            numeric_cols = df_copy.select_dtypes(include=['float64', 'int64']).columns
            df_copy[numeric_cols] = self.scaler.transform(df_copy[numeric_cols])
            
        return df_copy
        
    def split_train_test(self, df, target_col, test_size=0.2, random_state=42):
        """分割训练集和测试集"""
        X = df.drop(target_col, axis=1) if target_col in df.columns else df
        y = df[target_col] if target_col in df.columns else None
        
        if y is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            return X_train, X_test, y_train, y_test
        else:
            # 如果没有目标列，只返回特征集
            return X, None, None, None