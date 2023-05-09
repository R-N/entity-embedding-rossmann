import numpy
numpy.random.seed(123)
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn import neighbors
from sklearn.preprocessing import Normalizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.layers import Input, Dense, Activation, Reshape
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Embedding
from tensorflow.keras.callbacks import ModelCheckpoint

#%%
import pickle


def embed_features(X, saved_embeddings_fname):
    # f_embeddings = open("embeddings_shuffled.pickle", "rb")
    f_embeddings = open(saved_embeddings_fname, "rb")
    embeddings = pickle.load(f_embeddings)

    index_embedding_mapping = {1: 0, 2: 1, 4: 2, 5: 3, 6: 4, 7: 5}
    X_embedded = []

    (num_records, num_features) = X.shape
    for record in X:
        embedded_features = []
        for i, feat in enumerate(record):
            feat = int(feat)
            if i not in index_embedding_mapping.keys():
                embedded_features += [feat]
            else:
                embedding_index = index_embedding_mapping[i]
                embedded_features += embeddings[embedding_index][feat].tolist()

        X_embedded.append(embedded_features)

    return numpy.array(X_embedded)




class Model(object):

    def evaluate(self, X_val, y_val):
        assert(min(y_val) > 0)
        guessed_sales = self.guess(X_val)
        relative_err = numpy.absolute((y_val - guessed_sales) / y_val)
        result = numpy.sum(relative_err) / len(y_val)
        return result


class LinearModel(Model):

    def __init__(self, X_train, y_train, X_val, y_val):
        super().__init__()
        self.clf = linear_model.LinearRegression()
        self.clf.fit(X_train, numpy.log(y_train))
        print("Result on validation data: ", self.evaluate(X_val, y_val))

    def guess(self, feature):
        return numpy.exp(self.clf.predict(feature))


class RF(Model):

    def __init__(self, X_train, y_train, X_val, y_val):
        super().__init__()
        self.clf = RandomForestRegressor(n_estimators=200, verbose=True, max_depth=35, min_samples_split=2,
                                         min_samples_leaf=1)
        self.clf.fit(X_train, numpy.log(y_train))
        print("Result on validation data: ", self.evaluate(X_val, y_val))

    def guess(self, feature):
        return numpy.exp(self.clf.predict(feature))


class SVM(Model):

    def __init__(self, X_train, y_train, X_val, y_val):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.__normalize_data()
        self.clf = SVR(kernel='linear', degree=3, gamma='auto', coef0=0.0, tol=0.001,
                       C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)

        self.clf.fit(self.X_train, numpy.log(self.y_train))
        print("Result on validation data: ", self.evaluate(X_val, y_val))

    def __normalize_data(self):
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)

    def guess(self, feature):
        return numpy.exp(self.clf.predict(feature))


class XGBoost(Model):

    def __init__(self, X_train, y_train, X_val, y_val):
        super().__init__()
        dtrain = xgb.DMatrix(X_train, label=numpy.log(y_train))
        evallist = [(dtrain, 'train')]
        param = {'nthread': -1,
                 'max_depth': 7,
                 'eta': 0.02,
                 'silent': 1,
                 'objective': 'reg:linear',
                 'colsample_bytree': 0.7,
                 'subsample': 0.7}
        num_round = 3000
        self.bst = xgb.train(param, dtrain, num_round, evallist)
        print("Result on validation data: ", self.evaluate(X_val, y_val))

    def guess(self, feature):
        dtest = xgb.DMatrix(feature)
        return numpy.exp(self.bst.predict(dtest))


class HistricalMedian(Model):

    def __init__(self, X_train, y_train, X_val, y_val):
        super().__init__()
        self.history = {}
        self.feature_index = [1, 2, 3, 4]
        for x, y in zip(X_train, y_train):
            key = tuple(x[self.feature_index])
            self.history.setdefault(key, []).append(y)
        print("Result on validation data: ", self.evaluate(X_val, y_val))

    def guess(self, features):
        features = numpy.array(features)
        features = features[:, self.feature_index]
        guessed_sales = [numpy.median(self.history[tuple(feature)]) for feature in features]
        return numpy.array(guessed_sales)


class KNN(Model):

    def __init__(self, X_train, y_train, X_val, y_val):
        super().__init__()
        self.normalizer = Normalizer()
        self.normalizer.fit(X_train)
        self.clf = neighbors.KNeighborsRegressor(n_neighbors=10, weights='distance', p=1)
        self.clf.fit(self.normalizer.transform(X_train), numpy.log(y_train))
        print("Result on validation data: ", self.evaluate(self.normalizer.transform(X_val), y_val))

    def guess(self, feature):
        return numpy.exp(self.clf.predict(self.normalizer.transform(feature)))

class NN_with_EntityEmbedding(Model):

    def __init__(
        self, 
        features=[
            ("store", 54, 10),
            ("dow", 7, 6),
            ("promo", 0, 1),
            ("year", 3, 2),
            ("month", 12, 6),
            ("day", 31, 10),
            ("state", 9, 6),
            #("store_type", 5, 3),
            #("city", 22, 10),
            #("family", 33, 10),
        ],
        model_dims=[1000, 500],
        model_init="uniform",
        model_activation="relu"
    ):
        super().__init__()
        
        self.features = features
        self.feature_count = len(self.features)
        self.model_dims = model_dims
        self.model_init = model_init
        self.model_activation = model_activation

        self.checkpointer = ModelCheckpoint(filepath="best_model_weights.hdf5", verbose=1, save_best_only=True)
        self.__build_keras_model()

    def split_features(self, X):
        X_list = [
            X[..., i]
            for i in range(self.feature_count)
        ]
        return X_list

    def preprocessing(self, X):
        X_list = self.split_features(X)
        return X_list

    def __build_keras_model(self):
        input_model = [Input(shape=(1,)) for i in range(self.feature_count)]
        output_embeddings = [
            Reshape(target_shape=(hidden_dim,))(
                Embedding(input_dim, hidden_dim, name=f"{f}_embedding")(input)
            ) if input_dim else Dense(hidden_dim)(input)
            for input, (f, input_dim, hidden_dim) in zip(input_model, self.features)
        ]

        output_model = Concatenate()(output_embeddings)
        for dim in self.model_dims:
            output_model = Dense(dim, kernel_initializer=self.model_init)(output_model)
            output_model = Activation(self.model_activation)(output_model)
        output_model = Dense(1)(output_model)
        output_model = Activation('sigmoid')(output_model)

        self.model = KerasModel(inputs=input_model, outputs=output_model)

        self.model.compile(loss='mean_absolute_error', optimizer='adam')

    def _val_for_fit(self, val):
        val = numpy.log(val) / self.max_log_y
        return val

    def _val_for_pred(self, val):
        return numpy.exp(val * self.max_log_y)

    def fit(self, X_train, y_train, X_val, y_val, epochs=10):
        self.max_log_y = max(numpy.max(numpy.log(y_train)), numpy.max(numpy.log(y_val)))
        self.model.fit(self.preprocessing(X_train), self._val_for_fit(y_train),
                       validation_data=(self.preprocessing(X_val), self._val_for_fit(y_val)),
                       epochs=epochs, batch_size=128,
                       # callbacks=[self.checkpointer],
                       )
        # self.model.load_weights('best_model_weights.hdf5')
        print("Result on validation data: ", self.evaluate(X_val, y_val))

    def guess(self, features):
        features = self.preprocessing(features)
        result = self.model.predict(features).flatten()
        return self._val_for_pred(result)


class NN(Model):

    def __init__(self, 
        X_train, y_train, 
        X_val, y_val,
        epochs=10,
        model_dims=[1000, 500],
        model_init="uniform",
        model_activation="relu"
    ):
        super().__init__()
        self.epochs = epochs

        self.model_dims = model_dims
        self.model_init = model_init
        self.model_activation = model_activation

        self.checkpointer = ModelCheckpoint(filepath="best_model_weights.hdf5", verbose=1, save_best_only=True)
        self.max_log_y = max(numpy.max(numpy.log(y_train)), numpy.max(numpy.log(y_val)))
        self.__build_keras_model()
        self.fit(X_train, y_train, X_val, y_val)

    def __build_keras_model(self):
        self.model = Sequential()
        for dim in self.model_dims:
            output_model = Dense(dim, kernel_initializer=self.model_init)(output_model)
            output_model = Activation(self.model_activation)(output_model)
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))

        self.model.compile(loss='mean_absolute_error', optimizer='adam')

    def _val_for_fit(self, val):
        val = numpy.log(val) / self.max_log_y
        return val

    def _val_for_pred(self, val):
        return numpy.exp(val * self.max_log_y)

    def fit(self, X_train, y_train, X_val, y_val):
        self.model.fit(X_train, self._val_for_fit(y_train),
                       validation_data=(X_val, self._val_for_fit(y_val)),
                       epochs=self.epochs, batch_size=128,
                       # callbacks=[self.checkpointer],
                       )
        # self.model.load_weights('best_model_weights.hdf5')
        print("Result on validation data: ", self.evaluate(X_val, y_val))

    def guess(self, features):
        result = self.model.predict(features).flatten()
        return self._val_for_pred(result)
