import pandas as pd
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import os

# 设置设备为CPU
device = torch.device('cpu')

# 设置中文显示
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

# 数据加载与预处理
def load_and_preprocess_data():
    # 加载附件一数据
    base_info = pd.read_csv('附件1：123家有信贷记录企业的相关数据.xlsx_0.csv')
    seller_invoices = pd.read_csv('附件1：123家有信贷记录企业的相关数据.xlsx_1.csv')
    buyer_invoices = pd.read_csv('附件1：123家有信贷记录企业的相关数据.xlsx_2.csv')

    # 数据清洗
    seller_invoices = seller_invoices[seller_invoices['发票状态'] == '有效发票']
    buyer_invoices = buyer_invoices[buyer_invoices['发票状态'] == '有效发票']

    # 特征工程 - 聚合发票数据
    def aggregate_invoice_data(invoice_df, prefix):
        agg_features = invoice_df.groupby('企业代号').agg({
            '金额': ['sum', 'mean', 'std', 'max', 'min'],
            '税额': ['sum', 'mean', 'std', 'max', 'min'],
            '价税合计': ['sum', 'mean', 'std', 'max', 'min'],
            '发票号码': 'count'
        }).reset_index()
        agg_features.columns = [f'{prefix}_{col[0]}_{col[1]}' if col[1] else col[0] for col in agg_features.columns]
        return agg_features

    # 聚合销方和购方发票特征
    seller_features = aggregate_invoice_data(seller_invoices, 'seller')
    buyer_features = aggregate_invoice_data(buyer_invoices, 'buyer')

    # 合并特征
    train_data = base_info.merge(seller_features, on='企业代号', how='left')
    train_data = train_data.merge(buyer_features, on='企业代号', how='left')

    # 处理缺失值
    train_data = train_data.fillna(0)

    # 编码目标变量
    label_encoder = LabelEncoder()
    train_data['信誉评级'] = label_encoder.fit_transform(train_data['信誉评级'])

    # 选择特征和目标
    X = train_data.drop(['企业代号', '企业名称', '信誉评级', '是否违约'], axis=1)
    y = train_data['信誉评级']

    # 特征标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 转换为PyTorch张量
    X_train = torch.FloatTensor(X_train).to(device)
    X_val = torch.FloatTensor(X_val).to(device)
    y_train = torch.LongTensor(y_train.values).to(device)
    y_val = torch.LongTensor(y_val.values).to(device)

    return X_train, X_val, y_train, y_val, label_encoder, scaler, X.columns

# 构建神经网络模型
class CreditRatingModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CreditRatingModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim//2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim//2, output_dim)
        )

    def forward(self, x):
        return self.layers(x)

# 训练模型
def train_model(X_train, y_train, X_val, y_val, input_dim):
    model = CreditRatingModel(input_dim, 128, 4).to(device)  # 4个评级类别，移至CPU
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # 训练参数
    epochs = 100
    batch_size = 32
    best_val_loss = float('inf')
    best_model = None

    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 训练循环
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        # 计算平均训练损失
        train_loss /= len(train_loader.dataset)

        # 在验证集上评估
        model.eval()
        with torch.no_grad():
            outputs = model(X_val)
            val_loss = criterion(outputs, y_val).item()
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == y_val).sum().item() / len(y_val)

        # 学习率调度
        scheduler.step(val_loss)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()

        # 打印进度
        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')

    # 加载最佳模型
    model.load_state_dict(best_model)
    return model

# 预测附件二企业信誉评级
def predict_credit_ratings(model, label_encoder, scaler, feature_names):
    # 加载附件二数据
    test_base = pd.read_csv('附件2：302家无信贷记录企业的相关数据.xlsx_0.csv')
    try:
        test_seller = pd.read_csv('附件2：302家无信贷记录企业的相关数据.xlsx_1.csv')
        test_buyer = pd.read_csv('附件2：302家无信贷记录企业的相关数据.xlsx_2.csv')
    except:
        # 如果读取失败，使用空数据
        test_seller = pd.DataFrame(columns=['企业代号', '金额', '税额', '价税合计', '发票状态'])
        test_buyer = pd.DataFrame(columns=['企业代号', '金额', '税额', '价税合计', '发票状态'])

    # 数据清洗
    if not test_seller.empty:
        test_seller = test_seller[test_seller['发票状态'] == '有效发票']
    if not test_buyer.empty:
        test_buyer = test_buyer[test_buyer['发票状态'] == '有效发票']

    # 聚合特征
    def aggregate_test_data(invoice_df, prefix):
        if invoice_df.empty:
            agg_features = pd.DataFrame({'企业代号': test_base['企业代号']})
            for col in feature_names:
                if col.startswith(prefix):
                    agg_features[col] = 0
            return agg_features

        agg_features = invoice_df.groupby('企业代号').agg({
            '金额': ['sum', 'mean', 'std', 'max', 'min'],
            '税额': ['sum', 'mean', 'std', 'max', 'min'],
            '价税合计': ['sum', 'mean', 'std', 'max', 'min'],
            '发票号码': 'count'
        }).reset_index()
        agg_features.columns = [f'{prefix}_{col[0]}_{col[1]}' if col[1] else col[0] for col in agg_features.columns]
        return agg_features

    seller_test = aggregate_test_data(test_seller, 'seller')
    buyer_test = aggregate_test_data(test_buyer, 'buyer')

    # 合并特征
    test_data = test_base.merge(seller_test, on='企业代号', how='left')
    test_data = test_data.merge(buyer_test, on='企业代号', how='left')

    # 处理缺失值
    test_data = test_data.fillna(0)

    # 选择特征
    X_test = test_data[feature_names]
    X_test_scaled = scaler.transform(X_test)

    # 转换为PyTorch张量并预测
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs.data, 1)

    # 解码预测结果
    test_data['信誉评级预测'] = label_encoder.inverse_transform(predicted)

    # 保存结果
    result = test_data[['企业代号', '企业名称', '信誉评级预测']]
    result.to_csv('信誉评级预测结果.csv', index=False)
    print('预测结果已保存至 信誉评级预测结果.csv')
    return result

# 主函数
def main():
    # 加载和预处理数据
    X_train, X_val, y_train, y_val, label_encoder, scaler, feature_names = load_and_preprocess_data()
    print(f'训练数据特征数量: {X_train.shape[1]}')

    # 训练模型
    model = train_model(X_train, y_train, X_val, y_val, X_train.shape[1])

    # 预测附件二企业信誉评级
    predictions = predict_credit_ratings(model, label_encoder, scaler, feature_names)
    print(predictions.head())

if __name__ == '__main__':
    main()