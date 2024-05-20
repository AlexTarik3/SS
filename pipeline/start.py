from train_model import train_model
from test_model import test_model


print('\nrandom_forest:')
train_model(file_name="train.csv",model_name='random_forest')
test_model(file_name='new_input.csv',model_name='random_forest')

print('\nxgboost:')
train_model(file_name="train.csv",model_name='xgboost')
test_model(file_name='new_input.csv',model_name='xgboost')

print('\nextra_trees:')
train_model(file_name="train.csv",model_name='extra_trees')
test_model(file_name='new_input.csv',model_name='extra_trees')