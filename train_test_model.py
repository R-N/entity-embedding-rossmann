import pickle
import numpy
numpy.random.seed(123)
from models import *
from sklearn.preprocessing import OneHotEncoder
import sys
sys.setrecursionlimit(10000)

train_ratio = 0.9
shuffle_data = False
one_hot_as_input = False
embeddings_as_input = False
save_embeddings = True
saved_embeddings_fname = "embeddings.pickle"  # set save_embeddings to True to create this file

f = open('feature_train_data.pickle', 'rb')
(X, y) = pickle.load(f)

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
    X = embed_features(X, saved_embeddings_fname)

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

# Load LabelEncoders
with open("les.pickle", 'rb') as f:
    les = pickle.load(f)

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

# print("Fitting NN...")
# for i in range(5):
#     models.append(NN(X_train, y_train, X_val, y_val))

# print("Fitting RF...")
# models.append(RF(X_train, y_train, X_val, y_val))

# print("Fitting KNN...")
# models.append(KNN(X_train, y_train, X_val, y_val))

# print("Fitting XGBoost...")
# models.append(XGBoost(X_train, y_train, X_val, y_val))


if save_embeddings:
    model = models[0].model
    embeddings = {f: e.get_weights()[0] for f, e in model.embedding_dict.items()}
    with open(saved_embeddings_fname, 'wb') as f:
        pickle.dump(embeddings, f, -1)


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
