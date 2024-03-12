import pandas as pd
import csv

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

'''
1. numpy 형식으로 바꾸고
2. 3차원 데이터로 바꾸기
'''
