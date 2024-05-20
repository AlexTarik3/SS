import pandas as pd
import pickle
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from preprocessing import preprocess_train_data

def train_model(file_name: str = 'train.csv', model_name: str = 'xgboost'):
    # loading data
    data = pd.read_csv('C:/Users/Omen/Desktop/СС/PR5/data/' + file_name)

    # preprocessing data
    data = preprocess_train_data(data)

    # split data
    X = data.drop(columns=['AdoptionSpeed'])
    Y = data['AdoptionSpeed']

    # models
    models = {
        'xgboost': XGBClassifier(verbosity=0),
        'random_forest': RandomForestClassifier(verbose=0),
        'extra_trees': ExtraTreesClassifier(verbose=0),
        'lightgbm': LGBMClassifier(verbose=0),
    }

    # training model
    model = models[model_name]
    model.fit(X, Y)

    # saving model
    with open(f'C:/Users/Omen/Desktop/СС/PR5/models/{model_name}.pkl', 'wb') as f:
        pickle.dump(model, f)