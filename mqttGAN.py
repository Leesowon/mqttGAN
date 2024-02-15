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
for key in data.keys():
    print(f"Key: {key}")
    print(data_npz[key])

# 특정 배열에 접근
specific_array = data_npz['xmal']  # 배열 이름에 따라 변경
print("Specific Array:")
print(specific_array)

# 어떤 데이터가 있는지 확인
for k in data_npz.files:
    print(f"{k}: {data_npz[k].shape}")
'''


# csv to npz
data_csv = pd.read_csv('D:/workspace/GAN/swGAN/data/mqttdataset_reduced.csv')


'''
my_data = np.genfromtxt(data_csv, delimiter=',')
np.save('my_data.npy', my_data)

# csv 용량이 클 때
df_csv = pd.read_csv(data_csv, delimiter=',')
df_csv = df_csv.to_numpy()
np.save('my_data', df_csv)

print('done')
'''


# 각 열을 NumPy 배열로 변환
arrays_dict = {column: data_csv[column].values for column in data_csv.columns}

# 딕셔너리 확인
print(arrays_dict)

# 각 키에 대한 데이터 크기 출력
for key, array in arrays_dict.items():
    print(f"{key}의 데이터 크기: {len(array)}")
    
'''



# CSV 파일 로드
with open('D:/workspace/GAN/swGAN/data/mqttdataset_reduced.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    data_dict = {key: [] for key in reader.fieldnames}
    for row in reader:
        for key in row.keys():
            if row[key] == '':
                # 기본값 설정
                data_dict[key].append(0)  # 또는 다른 값을 설정
            else:
                data_dict[key].append(float(row[key]))  # 또는 다른 형식으로 변환
                
# 딕셔너리의 각 값들을 NumPy 배열로 변환
arrays_dict = {key: np.array(value) for key, value in data_dict.items()}

# 결과 확인
print(arrays_dict)                

'''





# key 목록 확인
print("Keys:", arrays_dict.keys())

# key 갯수 확인
key_count = len(arrays_dict.keys())
print("Number of keys:", key_count)





# npz 파일 닫기 (필수적으로 닫아야 함)
data_npz.close()

'''
print("----------train70_reduced.csv 크기 확인----------")
# 데이터를 배열로 변환
data_array = data_csv.to_numpy()

# 배열의 크기 확인
array_shape = data_array.shape

# 결과 출력
print(f"CSV 파일을 배열로 변환한 크기: {array_shape}")
'''