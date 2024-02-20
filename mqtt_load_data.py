import numpy as np
import pandas as pd
import os
import csv

# csv 파일 읽기
data_csv = pd.read_csv('D:/workspace/GAN/swGAN/data/mqttdataset_reduced.csv')

# target 열에 대해 레이블 매핑 (ben : 0, mal : 1)
data_csv['label'] = data_csv['target'].map({'legitimate': 0, 'dos': 1, 'slowite': 1, 'bruteforce': 1, 'malformed': 1, 'flood': 1})

# 매핑 후 데이터를 CSV 파일로 저장
# data_csv.to_csv('D:\workspace\GAN\swGAN\data\origin_data_csv_mapping.csv', index=False)

# 정상과 악성 데이터를 분리
normal_data = data_csv[data_csv['label'] == 0]
malicious_data = data_csv[data_csv['label'] == 1]


# csv to npz

# 정상 데이터를 NumPy 배열로 변환
x_normal = normal_data.drop(['target', 'label'], axis=1).values
y_normal = normal_data['label'].values

# 악성 데이터를 NumPy 배열로 변환
x_mal = malicious_data.drop(['target', 'label'], axis=1).values
y_mal = malicious_data['label'].values

# 데이터 확인
print(x_mal)


# 바이너리 변환 안하고 그냥 .npz로 저장
np.savez('D:\workspace\GAN\swGAN\data\mqtt_data.npz', x_normal=x_normal, y_normal=y_normal, x_mal=x_mal, y_mal=y_mal)

mqtt_data_npz = np.load('D:\workspace\GAN\swGAN\data\mqtt_data.npz', allow_pickle=True)


# data 확인
for k in mqtt_data_npz.files:
    print(f"{k}: {mqtt_data_npz[k].shape}")


# npz 파일 닫기
# data_npz.close()
mqtt_data_npz.close()
