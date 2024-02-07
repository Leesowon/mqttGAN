import numpy as np

# npz 파일 읽기
data = np.load('D:/workspace/GAN/Malware-GAN-master/data.npz')

# 데이터 형식 확인
for k in data.files:
    print('--------')
    print(k)
    print('--------')
    
# npz 파일 내의 데이터 확인
for key in data.keys():
    print(f"Key: {key}")
    print(data[key])

# 특정 배열에 접근
specific_array = data['xmal']  # 배열 이름에 따라 변경
print("Specific Array:")
print(specific_array)

# npz 파일 닫기 (필수적으로 닫아야 함)
data.close()
