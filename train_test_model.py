import pickle
import numpy
numpy.random.seed(123)
from models import *
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
import sys
import pickle
from datetime import datetime
import numpy as np
import random
random.seed(42)
import pandas as pd
sys.setrecursionlimit(10000)

train_ratio = 0.9
shuffle_data = False
one_hot_as_input = False
embeddings_as_input = False
save_embeddings = True

features=[
    ("store", 54, 10),
    ("dow", 7, 6),
    ("promo", 0, 1),
    ("year", 3, 2),
    ("month", 12, 6),
    ("day", 31, 10),
    ("state", 16, 6),
    ("store_type", 5, 3),
    ("city", 22, 10),
    ("family", 33, 10),
]

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
for i in range(full_X.shape[1]):
    le = preprocessing.LabelEncoder()
    le.fit(full_X[:, i])
    les.append(le)
for i in range(train_data_X.shape[1]):
    train_data_X[:, i] = les[i].transform(train_data_X[:, i])

train_data_X = train_data_X.astype(int)
train_data_y = np.array(train_data_y)

(X, y) = train_data_X, train_data_y

with open('les.pickle', 'wb') as f:
    pickle.dump(les, f, -1)

le_dict = {
    f: le
    for le, (f, i, h) in zip(les, features)
}
with open('le_dict.pickle', 'wb') as f:
    pickle.dump(le_dict, f, -1)

num_records = len(X)
train_size = int(train_ratio * num_records)

if shuffle_data:
    print("Using shuffled data")
    sh = numpy.arange(X.shape[0])
    numpy.random.shuffle(sh)
    X = X[sh]
    y = y[sh]

if embeddings_as_input:
    print("Using learned embeddings as input")
    X = embed_features(X, "embeddings.pickle")

if one_hot_as_input:
    print("Using one-hot encoding as input")
    enc = OneHotEncoder(sparse=False)
    enc.fit(X)
    X = enc.transform(X)

X_train = X[:train_size]
X_val = X[train_size:]
y_train = y[:train_size]
y_val = y[train_size:]

def sample(X, y, n):
    '''random samples'''
    num_row = X.shape[0]
    indices = numpy.random.randint(num_row, size=n)
    return X[indices, :], y[indices]

n = 200000
n = min(n, train_size)
X_train, y_train = sample(X_train, y_train, n)  # Simulate data sparsity
print("Number of samples used for training: " + str(y_train.shape[0]))

features = [
    (f, len(les[i].classes_), hidden_dim)
    for i, (f, input_dim, hidden_dim) in enumerate(features)
]

models = []

print("Fitting NN_with_EntityEmbedding...")
for i in range(5):
    model = NN_with_EntityEmbedding(features=features)
    model.fit(
        X_train, y_train, 
        X_val, y_val,
        epochs=10
    )
    models.append(model)

if save_embeddings:
    model = models[0]
    embeddings = [e.get_weights()[0] for e in model.embeddings]
    with open("embeddings.pickle", 'wb') as f:
        pickle.dump(embeddings, f, -1)
    embeddings_dict = {f: e.get_weights()[0] for f, e in model.embedding_dict.items()}
    with open("embedding_dict.pickle", 'wb') as f:
        pickle.dump(embeddings_dict, f, -1)


def evaluate_models(models, X, y):
    assert(min(y) > 0)
    guessed_sales = numpy.array([model.guess(X) for model in models])
    mean_sales = guessed_sales.mean(axis=0)
    relative_err = numpy.absolute((y - mean_sales) / y)
    result = numpy.sum(relative_err) / len(y)
    return result


print("Evaluate combined models...")
print("Training error...")
r_train = evaluate_models(models, X_train, y_train)
print(r_train)

print("Validation error...")
r_val = evaluate_models(models, X_val, y_val)
print(r_val)
