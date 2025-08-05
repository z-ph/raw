import os
from chardet import detect

# 需要转换的目录列表
target_dirs = [
    r'd:\download\cumcm2020c\C\raw\源数据',
    # r'd:\download\cumcm2020c\C\raw\预处理数据',
]
target_encoding = 'utf-8-sig'

for dir_path in target_dirs:
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                
                # 检测原始编码
                with open(file_path, 'rb') as f:
                    raw = f.read()
                encoding = detect(raw)['encoding']
                
                # 转换为GBK
                try:
                    content = raw.decode(encoding).encode(target_encoding)
                    with open(file_path, 'wb') as f:
                        f.write(content)
                    print(f'成功转换：{file_path}')
                except Exception as e:
                    print(f'转换失败：{file_path}，错误：{str(e)}')