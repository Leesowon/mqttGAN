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
drop_null_data = df.drop(['mqtt.conack.flags', 'mqtt.conack.flags.reserved', 'mqtt.conack.flags.sp','mqtt.conflag.qos',
                          'mqtt.conflag.reserved', 'mqtt.conflag.retain', 'mqtt.conflag.willflag', 'mqtt.sub.qos', 'mqtt.suback.qos',
                          'mqtt.willmsg', 'mqtt.willmsg_len', 'mqtt.willtopic', 'mqtt.willtopic_len'], axis=1)
'''
'''
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

'''
print("xben.shape=", xben.shape) # (1000,19)
print("yben.shape=", yben.shape) # (1000,)
print("xmal.shape=", xmal.shape) # (1000,19)
print("ymal.shape=", ymal.shape) # (1000,)
# print("xmal[0]=", xmal.iloc[0])
'''
# print(type(xben)) # Dataframe
# print(type(xmal)) # Dataframe

# ('xben_columns : ', xben.columns)
# print('xmal_columns : ', xmal.columns)

# print(xben.shape[1]) # 열 갯수 : 19
# == len(xben.columns)

# 각 columns의 자료형만 확인
# print(xben.dtypes)


# dataframe -> csv -> numpy (x)
# xben_csv = xben.to_csv(header=False, index=False)
# xben = np.array([row.split(',') for row in xben_csv.strip().split('\n')])

# print('drop')
# xben = xben.drop(['tcp.flags', 'mqtt.hdrflags'], axis=1)
# xmal = xmal.drop(['tcp.flags', 'mqtt.hdrflags'], axis=1)
# print("xben.shape=", xben.shape) # (1000,17)

# print('xben data type : ', xben.dtypes)
# print('xmal data type : ', xmal.dtypes)


# hex to int
print(xben['tcp.flags'].dtype)
'''
print(xben['tcp.flags'][0])
integer_value = int(xben['tcp.flags'][0], 16)
print('변환 : ', integer_value)
'''

'''
# 0x로 시작하는 값 int로 반환 : 0x로 시작하지 않으면 반환x
def hex_to_int(value):
    if isinstance(value, str) and value.startswith('0x'):
        try:
            return int(value, 16)
        except ValueError:
            return value
    return value
    
# 0x로 시작하지 않는 16진수도 int로 반환 but 일반 int형도 반환해버림
def hex_to_int(value):
    if isinstance(value, str) and value.startswith('0x'):
        try:
            return int(value, 16)
        except ValueError:
            return value
    else:  # 0x로 시작하지 않는 경우
        try:
            return int(value, 16)
        except ValueError:
            return value

'''
'''
def hex_to_int(value):
    if isinstance(value, str):
        if value.startswith('0x'):
            try:
                return int(value, 16)
            except ValueError:
                return value
        else:
            try:
                # int_value = int(value)
                return int(value)
            except ValueError:
                return value
    else:  # 문자열이 아닌 경우 그대로 반환
        return value
'''
'''
# 16진수 -> int
def hex_to_int(value):
    if isinstance(value, str):
        if value.startswith('0x'):
            try:
                return int(value, 16)
            except ValueError:
                return value
        else:
            try:
                return int(value, 16)
            except ValueError:
                try:
                    return int(value)
                except ValueError:
                    try:
                        return float(value)
                    except ValueError:
                        return value
    else:  # 문자열이 아닌 경우 그대로 반환
        return value
'''
def hex_to_int(value):
    if isinstance(value, int):
        return value
    elif isinstance(value, str):
        if value.startswith('0x'):
            try:
                return int(value, 16)
            except ValueError:
                return value
        else:
            try:
                return int(value, 16)
            except ValueError:
                return value
    else:
        return value

def convert_hex_to_int(df):
    df['mqtt.protoname'] = df['mqtt.protoname'].map({'MQTT':1, '0':0})

    for column in df.columns:
        df[column] = df[column].apply(hex_to_int)
    return df

print('\n ### hex to int! ###\n')
xben = convert_hex_to_int(xben)
xmal = convert_hex_to_int(xmal)
print('\nxben.dtypes :\n', xben.dtypes)
print('\nxmal.dtypes :\n', xmal.dtypes)

'''
# print('\n ### header ###\n', xben['mqtt.hdrflags'].describe()) # int64
print(xmal['mqtt.conflags'].describe()) # float64
print(xmal['mqtt.hdrflags'].describe()) # float64
print(xmal['mqtt.msg'].describe()) # int 64
print(xmal['mqtt.protoname'].describe()) # int64
'''

# dataframe -> csv
print('\n### to_numpy() ###\n')
xben = xben.to_numpy()
xmal = xmal.to_numpy()
# print(xben)
# print(xben.shape)
# print(xmal.shape)

print('\n### astype(np.float32) ###\n')
xben = xben.astype(np.float64)
xmal = xmal.astype(np.float64)
# print(xben)
# print(xben.shape)
# print(xmal)
# print(xmal.shape)

print("\n##### X_test_ben max #####")
print(np.max(xben))
'''
print("\n##### X_test_ben min #####")
print(xben.min())

print("\n##### X_test_mal max #####")
print(xmal.max())

print("\n##### X_test_mal min #####")
print(xmal.min())

pd.DataFrame(xben).to_csv('D:\workspace\GAN\swGAN\data\\xben_noHex(24.03.19).csv')
pd.DataFrame(xmal).to_csv('D:\workspace\GAN\swGAN\data\\xmal_noHex(24.03.19).csv')
'''