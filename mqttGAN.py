import numpy as np
import pandas as pd
import csv

from sklearn.model_selection import train_test_split

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

print(ben.shape)
print(mal.shape)

ben = ben.drop(['tcp.time_delta', 'mqtt.conack.flags', 'mqtt.conack.flags.reserved', 'mqtt.conack.flags.sp', 'mqtt.conack.val', 'mqtt.conflag.cleansess',
          'mqtt.conflag.passwd', 'mqtt.conflag.qos', 'mqtt.conflag.reserved', 'mqtt.conflag.retain', 'mqtt.conflag.uname', 'mqtt.conflag.willflag',
          'mqtt.conflags', 'mqtt.dupflag', 'mqtt.kalive', 'mqtt.msgid', 'mqtt.msgtype', 'mqtt.proto_len', 'mqtt.protoname', 'mqtt.qos', 'mqtt.retain',
          'mqtt.sub.qos', 'mqtt.suback.qos', 'mqtt.ver', 'mqtt.willmsg', 'mqtt.willmsg_len', 'mqtt.willtopic', 'mqtt.willtopic_len', 'target'], axis=1)

mal = mal.drop(['tcp.time_delta', 'mqtt.conack.flags', 'mqtt.conack.flags.reserved', 'mqtt.conack.flags.sp', 'mqtt.conack.val', 'mqtt.conflag.cleansess',
          'mqtt.conflag.passwd', 'mqtt.conflag.qos', 'mqtt.conflag.reserved', 'mqtt.conflag.retain', 'mqtt.conflag.uname', 'mqtt.conflag.willflag',
          'mqtt.conflags', 'mqtt.dupflag', 'mqtt.kalive', 'mqtt.msgid', 'mqtt.msgtype', 'mqtt.proto_len', 'mqtt.protoname', 'mqtt.qos', 'mqtt.retain',
          'mqtt.sub.qos', 'mqtt.suback.qos', 'mqtt.ver', 'mqtt.willmsg', 'mqtt.willmsg_len', 'mqtt.willtopic', 'mqtt.willtopic_len', 'target'], axis=1)

print('drop:',ben.shape)
print('drop:',mal.shape)

print('##### 사용할 feature #####')
print('ben_columns : ', ben.columns)
print('mal_columns : ', mal.columns)

xben = ben
xmal = mal

# 시퀀스 한줄의 길이는 80
num_of_normal = 1000
num_of_mal = 1000

# xben = padding_data(f_ben, 'ben')
yben = np.zeros(num_of_normal)
# xmal = padding_data(f_mal, 'mal')
ymal = np.ones(num_of_mal)

print("xben.shape=", xben.shape)
print("yben.shape=", yben.shape)
print("xmal.shape=", xmal.shape)
print("ymal.shape=", ymal.shape)

print('\n##### xben, yben, xmal, ymal #####')
print("xben_data type : ", type(xben))
print("xben_data type : ", type(yben))
print("xben_data type : ", type(xmal))
print("xben_data type : ", type(ymal))

print('\n##### xben feature 별 데이터 타입 #####')
print('tcp.flags type : ', xben['tcp.flags'].dtype)
print('tcp.len type : ', xben['tcp.len'].dtype)
print('mqtt.hdrflags type : ', xben['mqtt.hdrflags'].dtype)
print('mqtt.len type : ', xben['mqtt.len'].dtype)
print('mqtt.msg type : ', xben['mqtt.msg'].dtype)


print('##### dataframe to numpy #####')
xben = xben.values
print(xben)
xmal = xmal.values
print(xmal)

print('xben nummpy? : ', type(xben))
print('xmal nummpy? : ', type(xmal))

train_size_a = 0.8

xtrain_mal, xtest_mal, ytrain_mal, ytest_mal = train_test_split(xmal, ymal, train_size=train_size_a,
                                                                    test_size=0.20, shuffle=False)
xtrain_ben, xtest_ben, ytrain_ben, ytest_ben = train_test_split(xben, yben, train_size=train_size_a,
                                                                    test_size=0.20, shuffle=False)

print("xtrain_mal.shape=", xtrain_mal.shape)
print("xtest_mal.shape=", xtest_mal.shape)
print("ytrain_mal.shape=", ytrain_mal.shape)
print("ytest_mal.shape=", ytest_mal.shape)