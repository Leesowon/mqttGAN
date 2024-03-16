import numpy as np
import pandas as pd
import csv

from sklearn.model_selection import train_test_split


def padding_data(data, data_type):
    # data = pd.read_csv(f)

    # print(data.head())

    # print("##### processing #####")
    # data = data.drop('target', axis=1)
    # print(data.head())

    # 데이터 프레임 열 개수 확인
    num = data.shape[1]  # 3
    # print('origin_data column : ', num)

    # 80개 열을 가진 데이터 프레임 생성
    new_data = pd.DataFrame()

    # new_line 생성
    for i in range(num):
        new_data[data.columns[i]] = data[data.columns[i]]

    for i in range(num, 80):
        new_data['new_col_' + str(i - num)] = 0  # 새로운 열에 0 채우기

    new_num = data.shape[1]  # 80
    # print(new_data)
    process_data = 'D:/workspace/GAN/swGAN/data/data_' + data_type + '_make_80_colums.csv'
    # print(file_name)

    # print('##### padding data size : ', new_data.shape, '#####')
    # 저장
    new_data.to_csv(process_data, index=False)
    return new_data

f_ben = "D:\\workspace\\GAN\\swGAN\\data\\mqttdataset_legitimate_1000.csv"
f_mal = "D:\\workspace\\GAN\\swGAN\\data\\mqttdataset_malicious_1000.csv"

'''
ben_csv = open(f_ben)
mal_csv = open(f_mal)

ben = csv.reader(f_ben)
mal = csv.reader(f_mal)

for row in ben:
    print(row)

line = ben_csv.readline()
print(line)
'''

# dataframe 형태로 불러오기
ben = pd.read_csv(f_ben)
mal = pd.read_csv(f_mal)

# print(ben.shape) # (1000,34)
# print(mal.shape) # (1000,34)

ben = ben.drop(['tcp.time_delta', 'mqtt.conack.flags', 'mqtt.conack.flags.reserved', 'mqtt.conack.flags.sp', 'mqtt.conack.val', 'mqtt.conflag.cleansess',
          'mqtt.conflag.passwd', 'mqtt.conflag.qos', 'mqtt.conflag.reserved', 'mqtt.conflag.retain', 'mqtt.conflag.uname', 'mqtt.conflag.willflag',
          'mqtt.conflags', 'mqtt.dupflag', 'mqtt.kalive', 'mqtt.msgid', 'mqtt.msgtype', 'mqtt.proto_len', 'mqtt.protoname', 'mqtt.qos', 'mqtt.retain',
          'mqtt.sub.qos', 'mqtt.suback.qos', 'mqtt.ver', 'mqtt.willmsg', 'mqtt.willmsg_len', 'mqtt.willtopic', 'mqtt.willtopic_len', 'target'], axis=1)

mal = mal.drop(['tcp.time_delta', 'mqtt.conack.flags', 'mqtt.conack.flags.reserved', 'mqtt.conack.flags.sp', 'mqtt.conack.val', 'mqtt.conflag.cleansess',
          'mqtt.conflag.passwd', 'mqtt.conflag.qos', 'mqtt.conflag.reserved', 'mqtt.conflag.retain', 'mqtt.conflag.uname', 'mqtt.conflag.willflag',
          'mqtt.conflags', 'mqtt.dupflag', 'mqtt.kalive', 'mqtt.msgid', 'mqtt.msgtype', 'mqtt.proto_len', 'mqtt.protoname', 'mqtt.qos', 'mqtt.retain',
          'mqtt.sub.qos', 'mqtt.suback.qos', 'mqtt.ver', 'mqtt.willmsg', 'mqtt.willmsg_len', 'mqtt.willtopic', 'mqtt.willtopic_len', 'target'], axis=1)

# print('drop missing data(ben):',ben.shape) # (1000,5)
# print('drop missing data(mal):',mal.shape) # (1000,5)

# print('##### 사용할 feature #####')
# print('ben_columns : ', ben.columns) # Index(['tcp.flags', 'tcp.len', 'mqtt.hdrflags', 'mqtt.len', 'mqtt.msg'], dtype='object')
# print('mal_columns : ', mal.columns)

xben = ben
xmal = mal

# 시퀀스 한줄의 길이는 80
num_of_normal = 1000
num_of_mal = 1000

# xben = padding_data(f_ben, 'ben')
yben = np.zeros(num_of_normal)
# xmal = padding_data(f_mal, 'mal')
ymal = np.ones(num_of_mal)

'''
print("xben.shape=", xben.shape) # (1000,5)
print("yben.shape=", yben.shape) # (1000,)
print("xmal.shape=", xmal.shape) # (1000,5)
print("ymal.shape=", ymal.shape) # (1000,)

print('\n##### xben, yben, xmal, ymal #####')
print("xben_data type : ", type(xben)) # <class 'pandas.core.frame.DataFrame'>
print("xben_data type : ", type(yben)) # <class 'numpy.ndarray'>
print("xben_data type : ", type(xmal)) # <class 'pandas.core.frame.DataFrame'>
print("xben_data type : ", type(ymal)) # <class 'numpy.ndarray'>

print('\n##### xben feature 별 데이터 타입 #####')
print('tcp.flags type : ', xben['tcp.flags'].dtype) # oject
print('tcp.len type : ', xben['tcp.len'].dtype) # float64
print('mqtt.hdrflags type : ', xben['mqtt.hdrflags'].dtype) # object
print('mqtt.len type : ', xben['mqtt.len'].dtype) # float64
print('mqtt.msg type : ', xben['mqtt.msg'].dtype) # int64
'''

# print('##### object to float #####')
# xben['tcp.flages'] = xben['tcp.flags'].astype(float)
# 문자열을 실수로 변환할 수 없음
# >> object 데이터 다 버리기


print('##### drop object features #####')
xben = xben.drop(['tcp.flags', 'mqtt.hdrflags'], axis=1)
xmal = xmal.drop(['tcp.flags', 'mqtt.hdrflags'], axis=1)

'''
print('xben_columns : ', ben.columns) # Index(['tcp.flags', 'tcp.len', 'mqtt.hdrflags', 'mqtt.len', 'mqtt.msg'], dtype='object')
print('xmal_columns : ', mal.columns) # Index(['tcp.flags', 'tcp.len', 'mqtt.hdrflags', 'mqtt.len', 'mqtt.msg'], dtype='object')

print('\n##### drop-object-xben feature 별 데이터 타입 #####')
print('tcp.len type : ', xben['tcp.len'].dtype) # float64
print('mqtt.len type : ', xben['mqtt.len'].dtype) # float64
print('mqtt.msg type : ', xben['mqtt.msg'].dtype) # int64
'''

'''
print('##### dataframe to numpy #####')
xben = xben.values
print(xben)
xmal = xmal.values
print(xmal)
'''

xben = padding_data(xben, 'ben')
yben = np.zeros(num_of_normal)
xmal = padding_data(xmal, 'mal')
ymal = np.ones(num_of_mal)

print('xben is nummpy or dataframe? : ', type(xben)) # <class 'pandas.core.frame.DataFrame'>
print('xmal is nummpy or dataframe? : ', type(xmal)) # <class 'pandas.core.frame.DataFrame'>

train_size_a = 0.8

xtrain_mal, xtest_mal, ytrain_mal, ytest_mal = train_test_split(xmal, ymal, train_size=train_size_a,
                                                                    test_size=0.20, shuffle=False)
xtrain_ben, xtest_ben, ytrain_ben, ytest_ben = train_test_split(xben, yben, train_size=train_size_a,
                                                                    test_size=0.20, shuffle=False)
'''
print("xtrain_mal.shape=", xtrain_mal.shape) # (800, 3)
print("xtest_mal.shape=", xtest_mal.shape) # (200,3)
print("ytrain_mal.shape=", ytrain_mal.shape) # (800,)
print("ytest_mal.shape=", ytest_mal.shape) # (200,)

print("X_train_mal type : ", type(xtrain_mal)) #  <class 'pandas.core.frame.DataFrame'>


print("\n##### X_test_ben max #####")
print(xtest_ben.max())

print("\n##### X_test_ben min #####")
print(xtest_ben.min())

print("\n##### X_test_mal max #####")
print(xtest_mal.max())

print("\n##### X_test_mal min #####")
print(xtest_mal.min())
'''

# dataframe -> numpy 변환 & reshape : numpy 배열의 형태 변경 / 데이터셋을 (800, 80)에서 (800, 1, 80)으로 변경
xtrain_mal = xtrain_mal.to_numpy().reshape((xtrain_mal.shape[0], 1) + xtrain_mal.shape[1:])
xtrain_ben = xtrain_ben.to_numpy().reshape((xtrain_ben.shape[0], 1) + xtrain_ben.shape[1:])
xtest_mal = xtest_mal.to_numpy().reshape((xtest_mal.shape[0], 1) + xtest_mal.shape[1:])
xtest_ben = xtest_ben.to_numpy().reshape((xtest_ben.shape[0], 1) + xtest_ben.shape[1:])
print('\ndone1(ben) : ', xtrain_ben.shape)
print('done1(mal) : ', xtrain_mal.shape)

# 데이터를 정규화하고 새로운 형태로 재구성
xtrain_mal = (xtrain_mal.astype(np.float64) - 35000.0) / 35000.0 # float32 -> 64
xtrain_mal = xtrain_mal.reshape((xtrain_mal.shape[0], 1) + xtrain_mal.shape[1:])
xtrain_ben = (xtrain_ben.astype(np.float64) - 35000.0) / 35000.0 # float32 -> 64
xtrain_ben = xtrain_ben.reshape((xtrain_ben.shape[0], 1) + xtrain_mal.shape[1:])
print('\ndone2(ben) : ', xtrain_ben.shape)
print('done2(mal) : ', xtrain_mal.shape)

xtest_mal = (xtest_mal.astype(np.float64) - 35000.0) / 35000.0 # float32 -> 64
xtest_mal = xtest_mal.reshape((xtest_mal.shape[0], 1) + xtest_mal.shape[1:])
xtest_ben = (xtest_ben.astype(np.float64) - 35000.0) / 35000.0 # float32 -> 64
xtest_ben = xtest_ben.reshape((xtest_ben.shape[0], 1) + xtest_ben.shape[1:])
print('\ndone3(ben) : ', xtrain_ben.shape)
print('done3(mal) : ', xtrain_mal.shape)

# 각 데이터셋을 (800, 1, 10, 8) 또는 (200, 1, 10, 8)로 reshape하여 4차원으로 변환
# (샘플 수, 채널 수, 높이, 너비)
xtrain_mal2 = xtrain_mal.reshape(xtrain_mal.shape[0], 1, 10, 8)
xtest_mal2 = xtest_mal.reshape(xtest_mal.shape[0], 1, 10, 8)
xtrain_ben2 = xtrain_ben.reshape(xtrain_ben.shape[0], 1, 10, 8)
xtest_ben2 = xtest_ben.reshape(xtest_ben.shape[0], 1, 10, 8)
print('\ndone final(ben) : ', xtrain_ben2.shape)
print('done final(mal) : ', xtrain_mal2.shape)

'''
xtrain_mal = (xtrain_mal.astype(np.float64) - 692.0) / 692.0
xtrain_mal = xtrain_mal.reshape((xtrain_mal.shape[0], 1) + xtrain_mal.shape[1:])
xtrain_ben = (xtrain_ben.astype(np.float64) - 692.0) / 692.0
xtrain_ben = xtrain_ben.reshape((xtrain_ben.shape[0], 1) + xtrain_mal.shape[1:])

xtest_mal = (xtest_mal.astype(np.float64) - 692.0) / 692.0
xtest_mal = xtest_mal.reshape((xtest_mal.shape[0], 1) + xtest_mal.shape[1:])
xtest_ben = (xtest_ben.astype(np.float64) - 692.0) / 692.0
xtest_ben = xtest_ben.reshape((xtest_ben.shape[0], 1) + xtest_ben.shape[1:])
'''

print('\n type')
print('xtrain_ben.type : ', xtrain_ben.dtype)
print('xtrain_mal.type : ', xtrain_mal.dtype)