import pandas as pd
import os

def load_preprocessed_data(file_path):
    """
    加载预处理后的企业特征数据
    
    参数:
    file_path: str - 预处理数据文件的绝对路径
    
    返回:
    pd.DataFrame - 包含企业特征数据的DataFrame
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件不存在: {file_path}")
        
    # 读取CSV文件，处理可能的编码问题
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='gbk')
        
    # 移除可能的无关列
    if '企业名称' in df.columns:
        df = df.drop('企业名称', axis=1)
        
    return df

def load_attachment1_data():
    """
    加载附件一(有信贷记录企业)的预处理数据
    
    返回:
    pd.DataFrame - 包含附件一企业特征数据的DataFrame
    """
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                            '预处理数据', '附件一企业特征_优化.csv')
    return load_preprocessed_data(data_path)

def load_attachment2_data():
    """
    加载附件二(无信贷记录企业)的预处理数据
    
    返回:
    pd.DataFrame - 包含附件二企业特征数据的DataFrame
    """
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                            '预处理数据', '附件二企业特征_优化.csv')
    return load_preprocessed_data(data_path)