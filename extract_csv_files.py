import pickle
import csv
import pandas as pd


def csv2dicts(csvfile):
    df = pd.read_csv(csvfile)
    df = df.fillna('')
    return df.to_dict('records')


def set_nan_as_string(data, replace_str='0'):
    for i, x in enumerate(data):
        for key, value in x.items():
            if value == '':
                x[key] = replace_str
        data[i] = x


train_data = "train.csv"
store_data = "store.csv"

data = csv2dicts(train_data)
with open('train_data.pickle', 'wb') as f:
    data = data[::-1]
    pickle.dump(data, f, -1)
    print(data[:3])


data = csv2dicts(store_data)
with open('store_data.pickle', 'wb') as f:
    pickle.dump(data, f, -1)
    print(data[:2])
