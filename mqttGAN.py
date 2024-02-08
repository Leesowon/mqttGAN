import numpy as np
import pandas as pd
import os

# npz 파일 읽기
data_npz = np.load('D:/workspace/GAN/Malware-GAN-master/data.npz')
'''
# npz 파일 내의 데이터 key 확인
for key in data.files:
    print(key) 
'''
'''
# npz 파일 내의 데이터 확인
for key in data.keys():
    print(f"Key: {key}")
    print(data[key])
'''
'''
# 특정 배열에 접근
specific_array = data['xmal']  # 배열 이름에 따라 변경
print("Specific Array:")
print(specific_array)
'''

# 어떤 데이터가 있는지 확인
for k in data_npz.files:
    print(f"{k}: {data_npz[k].shape}")

# 전체 파일 크기 확인
file_size_bytes = os.path.getsize(data_npz)
file_size_kb = file_size_bytes / 1024
file_size_mb = file_size_kb / 1024

print(f"File Size: {file_size_bytes} bytes ({file_size_kb:.2f} KB, {file_size_mb:.2f} MB)")


'''
# csv to npz
path_of_csv = 'D:/workspace/GAN/swGAN/data/train70_reduced.csv'

my_data = np.genfromtxt(path_of_csv, delimiter=',')
np.save('my_data.npy', my_data)

# csv 용량이 클 때
df_csv = pd.read_csv(path_of_csv, delimiter=',')
df_csv = df_csv.to_numpy()
np.save('my_data', df_csv)

print('done')
'''
# npz 파일 닫기 (필수적으로 닫아야 함)
data_npz.close()
