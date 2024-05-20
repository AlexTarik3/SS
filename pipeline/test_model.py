import pandas as pd
import pickle
from preprocessing import preprocess_testing_data

def test_model(file_name: str = 'new_input.csv', model_name: str = 'xgboost'):
    # loading data
    data = pd.read_csv('C:/Users/Omen/Desktop/СС/PR5/data/' + file_name)

    # preprocessing data
    data = preprocess_testing_data(data)

    # split data
    X = data.drop(columns=['AdoptionSpeed'])
    Y = data['AdoptionSpeed']

    # testing model
    with open(f'C:/Users/Omen/Desktop/СС/PR5/models/{model_name}.pkl', 'rb') as f:
        model = pickle.load(f)

    predictions = model.predict(X)

    # saving predictions
    pd.DataFrame(predictions).to_csv('C:/Users/Omen/Desktop/СС/PR5/data/predictions.csv', index=False)

    # printing accuracy of predictions
    accuracy = (predictions == Y).mean()
    print(f'Accuracy: {accuracy}')
    