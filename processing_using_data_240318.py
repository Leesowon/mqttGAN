import numpy as np
import pandas as pd

'''
f_data = "D:\workspace\GAN\swGAN\data\origin_data\mqttdataset_reduced.csv"

# data frame 형태로 불러오기
df = pd.read_csv(f_data)

print('df.shape : ', df.shape) # (330926, 34)
print('len(df) : ', len(df))
print('df.columns : ', df.columns)
print('type(df) : ', type(df)) # <class 'pandas.core.frame.DataFrame'>
print('\ndf.info()', df.info())

print('\n')
'''

# print('\n', df['tcp.flags'].unique())
# print('\n', df['tcp.time_delta'].unique())
# print('\n', df['tcp.len'].unique()) # pass
# print('\n', df['mqtt.conack.flags'].unique()) # drop
# print('\n', df['mqtt.conack.flags.reserved'].unique()) # drop
# print('\n', df['mqtt.conack.flags.sp'].unique()) # drop
# print('\n', df['mqtt.conack.val'].unique())
# print('\n', df['mqtt.conflag.cleansess'].unique())
# print('\n', df['mqtt.conflag.passwd'].unique())
# print('\n', df['mqtt.conflag.qos'].unique()) # drop
# print('\n', df['mqtt.conflag.reserved'].unique()) # drop
# print('\n', df['mqtt.conflag.retain'].unique()) # drop
# print('\n', df['mqtt.conflag.uname'].unique())
# print('\n', df['mqtt.conflag.willflag'].unique()) # drop
# print('\n', df['mqtt.conflags'].unique())
# print('\n', df['mqtt.dupflag'].unique())
# print('\n', df['mqtt.hdrflags'].unique())
# print('\n', df['mqtt.kalive'].unique())
# print('\n', df['mqtt.len'].unique())
# print('\n', df['mqtt.msg'].unique())
# print('\n', df['mqtt.msgid'].unique())
# print('\n', df['mqtt.msgtype'].unique())
# print('\n', df['mqtt.proto_len'].unique())
# print('\n', df['mqtt.protoname'].unique())
# print('\n', df['mqtt.qos'].unique())
# print('\n', df['mqtt.retain'].unique())
# print('\n', df['mqtt.sub.qos'].unique()) # drop
# print('\n', df['mqtt.suback.qos'].unique()) # drop
# print('\n', df['mqtt.ver'].unique())
# print('\n', df['mqtt.willmsg'].unique()) # drop
# print('\n', df['mqtt.willmsg_len'].unique()) # drop
# print('\n', df['mqtt.willtopic'].unique()) # drop
# print('\n', df['mqtt.willtopic_len'].unique()) # drop
# print('\n', df['target'].unique())
'''
# missing 데이터 제거
drop_null_data = df.drop(['mqtt.conack.flags', 'mqtt.conack.flags.reserved', 'mqtt.conack.flags.sp','mqtt.conflag.qos', 'mqtt.conflag.reserved',
                          'mqtt.conflag.retain', 'mqtt.conflag.willflag', 'mqtt.sub.qos', 'mqtt.suback.qos', 'mqtt.willmsg', 'mqtt.willmsg_len', 'mqtt.willtopic',
                          'mqtt.willtopic_len'], axis=1)
# 저장
# drop_null_data.to_csv('D:\workspace\GAN\swGAN\data\drop_null_data(24.03.16).csv', index=False)

# 저장 확인
f_drop_null_data = 'D:\workspace\GAN\swGAN\data\drop_null_data(24.03.16).csv'
df_dropNull_check = pd.read_csv(f_drop_null_data)
print('drop null data :',df_dropNull_check.shape) # (330926, 21)
'''
'''
# 정상 / 악성 데이터 분리
seperate_ben = df_dropNull[df_dropNull['target'] == 'legitimate']
seperate_ben.to_csv('D:\workspace\GAN\swGAN\data\drop_null_data_ben(24.03.18).csv', index=False)

seperate_mal = df_dropNull[df_dropNull['target'].isin(['dos', 'slowite', 'bruteforce', 'malformed', 'flood'])]
seperate_mal.to_csv('D:\workspace\GAN\swGAN\data\drop_null_data_mal(24.03.18).csv', index=False)


f_ben = 'D:\workspace\GAN\swGAN\data\drop_null_data_ben(24.03.18).csv'
f_mal = 'D:\workspace\GAN\swGAN\data\drop_null_data_mal(24.03.18).csv'

ben = pd.read_csv(f_ben)
mal = pd.read_csv(f_mal)

# print('ben.shape :',ben.shape) # (165463, 21)
# print('mal.shape :',mal.shape) # (165463, 21)


# labeling 될 target 제거, 시간 피쳐 제거
ben = ben.drop(['target', 'tcp.time_delta'], axis=1)
mal = mal.drop(['target', 'tcp.time_delta'], axis=1)

print('ben.shape :',ben.shape) # (165463, 19)
print('mal.shape :',mal.shape) # (165463, 19)
'''
# 1000개 데이터 랜덤 추출(sample) / 1000개의 데이터 비복원 추출
num_of_normal = 1000
num_of_mal = 1000
'''
xben = ben.sample(n = num_of_normal, replace = False)
yben = np.zeros(num_of_normal)
xmal = mal.sample(n = num_of_mal, replace = False)
ymal = np.zeros(num_of_mal)

# 저장
xben.to_csv('D:\\workspace\\GAN\\swGAN\\data\\xben(24.03.18).csv', index=False)
xmal.to_csv('D:\\workspace\\GAN\\swGAN\\data\\xmal(24.03.18).csv', index=False)
'''
f_xben = 'D:\\workspace\\GAN\\swGAN\\data\\xben(24.03.18).csv'
f_xmal = 'D:\\workspace\\GAN\\swGAN\\data\\xmal(24.03.18).csv'

xben = pd.read_csv(f_xben)
yben = np.zeros(num_of_normal)
xmal = pd.read_csv(f_xmal)
ymal = np.zeros(num_of_mal)

print("xben.shape=", xben.shape) # (1000,19)
print("yben.shape=", yben.shape) # (1000,)
print("xmal.shape=", xmal.shape) # (1000,19)
print("ymal.shape=", ymal.shape) # (1000,)
# print("xmal[0]=", xmal.iloc[0])

print(type(xben)) # Dataframe
# print(type(xmal)) # Dataframe


# Dataframe -> numpy
# astype(np.float32) & -1~1 사이의 값으로 스케일링 : 이런 정규화과정은 학습과정에서 수렴을 촉진하고, 더 빠르고 안정적인 학습을 가능하게 하는데 도움


# 각 피쳐별 데이터 타입 확인