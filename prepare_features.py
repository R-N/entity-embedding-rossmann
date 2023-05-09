import pickle
from datetime import datetime
from sklearn import preprocessing
import numpy as np
import random
random.seed(42)

import pickle
import csv
import pandas as pd

def csv2dicts(csvfile):
    df = pd.read_csv(csvfile)
    df = df.fillna('')
    return df.to_dict('records')

train_data = "train.csv"
store_data = "store.csv"

train_data = csv2dicts(train_data)
store_data = csv2dicts(store_data)
    
num_records = len(train_data)

def feature_list(record):
    dt = datetime.strptime(record['date'], '%Y-%m-%d')
    store_index = int(record['store_nbr'])
    year = dt.year
    month = dt.month
    day = dt.day
    day_of_week = int(dt.weekday())
    store = store_data[store_index-1]
    family = record['family']
    promo = record['onpromotion']

    return [
        #1,
        store_index,
        day_of_week,
        promo,
        year,
        month,
        day,
        store['state'],
        store['type'],
        store['city'],
        family,
    ]

train_data_X = []
train_data_y = []

for record in train_data:
    if record['sales']:
        sales = float(record['sales'])
        if sales:
            fl = feature_list(record)
            train_data_X.append(fl)
            train_data_y.append(sales)
print("Number of train datapoints: ", len(train_data_y))

print(min(train_data_y), max(train_data_y))

full_X = train_data_X
full_X = np.array(full_X)
train_data_X = np.array(train_data_X)
les = []
for i in range(train_data_X.shape[1]):
    le = preprocessing.LabelEncoder()
    le.fit(full_X[:, i])
    les.append(le)
    train_data_X[:, i] = le.transform(train_data_X[:, i])

with open('les.pickle', 'wb') as f:
    pickle.dump(les, f, -1)

train_data_X = train_data_X.astype(int)
train_data_y = np.array(train_data_y)

with open('feature_train_data.pickle', 'wb') as f:
    pickle.dump((train_data_X, train_data_y), f, -1)
    print(train_data_X[0], train_data_y[0])
