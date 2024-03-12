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
'''

# line = f_ben.readline()
# print(line)

ben = pd.read_csv(f_ben)
mal = pd.read_csv(f_mal)

