import pandas as pd


def processing_data(f, data_type):
    data = pd.read_csv(f)
    print(data.head())

    print("##### processing #####")

    data = data.drop('target', axis=1)
    print(data.head())

    # 데이터 프레임 열 개수 확인
    num = data.shape[1]  # 33
    print('origin_data column : ', num)

    # 80개 열을 가진 데이터 프레임 생성
    new_data = pd.DataFrame()

    # 'new_line' 생성
    for i in range(num):
        new_data[data.columns[i]] = data[data.columns[i]]

    for i in range(num, 80):
        new_data['new_col_' + str(i - num)] = 0  # 새로운 열에 0 채우기

    # new_data['label'] = 'legitimate'  # 마지막 열에 'label' 추가

    print(new_data)

    new_num = data.shape[1]  # 80
    print('data column : ', new_data)
    process_data = 'D:/workspace/GAN/swGAN/data/data_' + data_type + '_make_80_colums.csv'
    # print(file_name)

    new_data.to_csv(process_data, index=False)


if __name__ == '__main__':
    f_ben = "D:\\workspace\\GAN\\swGAN\\data\\mqttdataset_legitimate_1000.csv"
    f_mal = "D:\\workspace\\GAN\\swGAN\\data\\mqttdataset_malicious_1000.csv"

processing_data(f_ben, 'ben')
processing_data(f_mal, 'mal')

'''
ben_data = pd.read_csv(f_ben)
mal_data = pd.read_csv(f_mal)

# print(ben_data.head())
# print(mal_data.head())

    print("##### processing #####")

    ben_data = ben_data.drop('target', axis=1)
    mal_data = mal_data.drop('target', axis=1)

    # print(ben_data.head())
    # print(mal_data.head())

    # 데이터 프레임 열 개수 확인
    num = ben_data.shape[1]  # 33
    print('data column : ', num)

    # feature_num = 33

    # 80개 열을 가진 데이터 프레임 생성
    new_data = pd.DataFrame()

    # 'new_line' 생성
    for i in range(num):
        new_data[ben_data.columns[i]] = ben_data[ben_data.columns[i]]

    for i in range(num, 80):
        new_data['new_col_' + str(i - num)] = 0  # 새로운 열에 0 채우기

    # new_data['label'] = 'legitimate'  # 마지막 열에 'label' 추가

    print(new_data)

    new_num = ben_data.shape[1]  # 80
    print('data column : ', new_data)

    ben_data.to_csv('D:/workspace/GAN/swGAN/data/ben_data_labeling.csv', index=False)
    mal_data.to_csv('D:/workspace/GAN/swGAN/data/mal_data_labeling.csv', index=False)
'''