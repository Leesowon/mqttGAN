import numpy as np
import pandas as pd
import os
import csv

# npz 파일 읽기
data_npz = np.load('D:/workspace/GAN/Malware-GAN-master/data.npz')

'''
# npz 파일 내의 데이터 key 확인
for key in data_npz.files:
    print(key) 

# npz 파일 내의 데이터 확인
for key in data_npz.keys():
    print(f"Key: {key}")
    print(data_npz[key])

# 특정 배열에 접근
specific_array = data_npz['yben']  # 배열 이름에 따라 변경
print("Specific Array:")
print(specific_array)

# 어떤 데이터가 있는지 확인
for k in data_npz.files:
    print(f"{k}: {data_npz[k].shape}")
'''

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
'''
# 각 열을 NumPy 배열로 변환
arrays_dict = {column: data_csv[column].values for column in data_csv.columns}

# 딕셔너리 확인
print(arrays_dict)
'''

# 정상 데이터를 NumPy 배열로 변환
x_normal = normal_data.drop(['target', 'label'], axis=1).values
y_normal = normal_data['label'].values

# 악성 데이터를 NumPy 배열로 변환
x_mal = malicious_data.drop(['target', 'label'], axis=1).values
y_mal = malicious_data['label'].values

'''
# 데이터 확인
print(x_mal)

# 결과 확인
print("Normal Data:")
print("x_normal shape:", x_normal.shape)
print("y_normal shape:", y_normal.shape)

print("\nMalicious Data:")
print("x_mal shape:", x_mal.shape)
print("y_mal shape:", y_mal.shape)
'''

'''
# 각 키에 대한 데이터 크기 출력
for key, array in arrays_dict.items():
    print(f"{key}의 데이터 크기: {len(array)}")

# key 목록 확인
print("Keys:", arrays_dict.keys())

# key 갯수 확인
key_count = len(arrays_dict.keys())
print("Number of keys:", key_count)
'''


# 각 파일을 .npy로 저장

# 정상 데이터 바이너리로 저장
np.save('x_normal.npy', x_normal)
np.save('y_normal.npy', y_normal)

# 악성 데이터 바이너리로 저장
np.save('x_mal.npy', x_mal)
np.save('y_mal.npy', y_mal)


# 데이터를 바이너리로 변환하여 .npz로 저장
np.savez('D:\workspace\GAN\swGAN\data\mqtt_data.npz', x_normal=x_normal, y_normal=y_normal, x_mal=x_mal, y_mal=y_mal)

# 파일 확인
# mqttdata_npz = np.load('D:\workspace\GAN\swGAN\data\mqtt_data.npz')
mqttdata_npz = np.load('D:\workspace\GAN\swGAN\data\mqtt_data.npz', allow_pickle=True)

# 특정 배열에 접근
specific_array = mqttdata_npz['x_mal']  # 배열 이름에 따라 변경
print("Specific Array:")
print(specific_array)

# npz 파일 닫기 (필수적으로 닫아야 함)
data_npz.close()
mqttdata_npz.close()
