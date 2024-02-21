import numpy as np
import pandas as pd
import os
import csv
import hashlib
import sys

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
print('x_mal')
print(x_mal)


# 바이너리 변환 안하고 그냥 .npz로 저장
np.savez('D:\workspace\GAN\swGAN\data\mqtt_data.npz', x_normal=x_normal, y_normal=y_normal, x_mal=x_mal, y_mal=y_mal)

mqtt_data_npz = np.load('D:\workspace\GAN\swGAN\data\mqtt_data.npz', allow_pickle=True)

# data 형식 확인
for k in mqtt_data_npz.files:
    print(f"{k}: {mqtt_data_npz[k].shape}")
    
# .npz 파일에 저장된 각 배열의 데이터 형식 확인
# for array_name, array_data in mqtt_data_npz.items():
#    print(f"{array_name}: {array_data.dtype}")
    
    '''
# .npz 파일에 저장된 각 배열의 데이터 형식 확인
for array_name, array_data in mqtt_data_npz.items():
    print(f"{array_name}: {array_data.dtype}")
    if array_name in ['x_normal', 'x_mal']:
        # x_normal, x_mal의 각 열에 대한 데이터 형식 확인
        for i in range(array_data.shape[1]):
            print(f"{array_name} - Column {i}: {array_data[:, i].dtype}")
'''

'''
# String, 실수.. data를 숫자로 변환
def process_mqtt_data(mqtt_data_npz):
    processed_data = []

    for row in mqtt_data_npz:
        processed_row = []
        for element in row:
            try:
                if '0x' in element:  # 16진수 문자열인 경우
                    processed_row.append(int(element, 16))
                else:  # 실수 값인 경우
                    processed_row.append(int(float(element)))
            except (ValueError, TypeError):  # 다른 형식의 데이터인 경우
                processed_row.append(0)  # 해당 데이터를 0으로 처리
        processed_data.append(processed_row)

    return np.array(processed_data, dtype=int)
'''

def hash_str_to_int(input_str):
    # 해시 함수의 반환 값을 np.iinfo(np.int64).max로 나눈 나머지를 사용
    return int(hashlib.sha256(input_str.encode()).hexdigest(), 16) % np.iinfo(np.int64).max

def process_mqtt_data(mqtt_data_npz):
    processed_data = []

    max_len = max(len(row) for row in mqtt_data_npz)  # 가장 긴 행의 길이를 찾습니다.

    for row in mqtt_data_npz:
        processed_row = []
        for element in row:
            try:
                if isinstance(element, str):  # 문자열인 경우
                    if '0x' in element:  # 16진수 문자열인 경우
                        processed_row.append(int(element, 16))
                    else:  # 일반 문자열인 경우
                        processed_row.append(hash_str_to_int(element))
                else:  # 실수 값인 경우
                    processed_row.append(int(element))
            except (ValueError, TypeError):  # 다른 형식의 데이터인 경우
                processed_row.append(0)  # 해당 데이터를 0으로 처리

        # 행의 길이가 max_len보다 짧으면, 0으로 채웁니다.
        if len(processed_row) < max_len:
            processed_row += [0] * (max_len - len(processed_row))

        processed_data.append(processed_row)

    return np.array(processed_data, dtype=np.int64)


# 데이터 처리 (y는 레이블이라 할 필요 x)
x_normal_processed = process_mqtt_data(mqtt_data_npz['x_normal'])
x_mal_processed = process_mqtt_data(mqtt_data_npz['x_mal'])

# 변환된 numpy들을 .npz 파일로 저장
np.savez('D:/workspace/GAN/swGAN/data/processed_mqtt_data.npz', 
         x_normal_processed=x_normal_processed, 
         y_normal=mqtt_data_npz['y_normal'],  # 변환하지 않고 그대로 저장
         x_mal_processed=x_mal_processed, 
         y_mal=mqtt_data_npz['y_mal'])  # 변환하지 않고 그대로 저장



print("x_normal_processed:")
print(x_normal_processed)

print("x_mal_processed:")
print(x_mal_processed)



# 저장된 .npz 데이터의 각각의 배열 크기 확인
saved_npz = np.load('D:/workspace/GAN/swGAN/data/processed_mqtt_data.npz', allow_pickle=True)
for k in saved_npz.files:
    print(f"{k}: {saved_npz[k].shape}")



# npz 파일 닫기
# data_npz.close()
mqtt_data_npz.close()
