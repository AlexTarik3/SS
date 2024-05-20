import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.simplefilter('ignore')

def preprocess_train_data(data: pd.DataFrame) -> pd.DataFrame:
    # dropping unnecessary columns
    data = data.drop(columns=['Unnamed: 0','BreedName_y','BreedName_x','Name','StateName_x',"Type",'PetID','RescuerID','Description','ColorName_x','ColorName_y','ColorName'],axis=1)

    # handling missing values using mode
    missing_cols = sorted(data.columns[data.isnull().any()].tolist(), key=lambda col: data[col].isnull().sum(), reverse=True)
    modes = {col: data[col].mode()[0] for col in data.columns}
    for col in missing_cols:
        data[col] = data[col].fillna(modes[col])

    # saving modes in a file
    with open('C:/Users/Omen/Desktop/СС/PR5/pipeline/settings/modes.pkl', 'wb') as f:
        pickle.dump(modes, f)

    categorical_cols = data.select_dtypes(include='object').columns

    one_hot_cols = ['MaturitySize',"Gender","FurLength","Vaccinated","Dewormed","Sterilized","Health"]

    ohe = OneHotEncoder(sparse=False)

    for col in one_hot_cols:
        data[col] = data[col].astype('str')

    ohe.fit(data[one_hot_cols])
    ohe_output = ohe.transform(data[one_hot_cols])
    ohe_output = pd.DataFrame(ohe_output)
    ohe_output.columns = ohe.get_feature_names_out(one_hot_cols)

    data = data.drop(one_hot_cols, axis=1)

    # replacing categorical columns with encoded columns
    data = data.drop(columns=categorical_cols)
    data = pd.concat([data, ohe_output], axis=1)

    # encoding categorical columns
    label_cols = [col for col in categorical_cols if len(data[col].unique()) == 2]

    label_encoders = {col: LabelEncoder() for col in label_cols}
    label_encoded = data[label_cols].apply(lambda col: label_encoders[col.name].fit_transform(col))

    # saving encoders in a file
    with open('C:/Users/Omen/Desktop/СС/PR5/pipeline/settings/onehot_encoder.pkl', 'wb') as f:
        pickle.dump(ohe, f)
    with open('C:/Users/Omen/Desktop/СС/PR5/pipeline/settings/label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    

    # replacing categorical columns with encoded columns
    data = data.drop(columns=categorical_cols)
    data = pd.concat([data, label_encoded], axis=1)

    # scaling numerical columns
    numerical_cols = ['Age', 'Fee', 'Quantity']

    scaler = MinMaxScaler()

    scaled = scaler.fit_transform(data[numerical_cols])
    data = data.drop(columns=numerical_cols)
    data = pd.concat([data, pd.DataFrame(scaled, columns=numerical_cols)], axis=1)

    # saving scaler in a file
    with open('C:/Users/Omen/Desktop/СС/PR5/pipeline/settings/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    return data

def preprocess_testing_data(data: pd.DataFrame) -> pd.DataFrame:
    # dropping unnecessary columns
    data = data.drop(columns=['Unnamed: 0','BreedName_y','BreedName_x','Name','StateName_x',"Type",'PetID','RescuerID','Description','ColorName_x','ColorName_y','ColorName'],axis=1)

    # handling missing values using mode
    missing_cols = sorted(data.columns[data.isnull().any()].tolist(), key=lambda col: data[col].isnull().sum(), reverse=True)
    with open('C:/Users/Omen/Desktop/СС/PR5/pipeline/settings/modes.pkl', 'rb') as f:
        modes = pickle.load(f)

    for col in missing_cols:
        data[col] = data[col].fillna(modes[col])

    # encoding categorical columns
    with open('C:/Users/Omen/Desktop/СС/PR5/pipeline/settings/label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    with open('C:/Users/Omen/Desktop/СС/PR5/pipeline/settings/onehot_encoder.pkl', 'rb') as f:
        ohe = pickle.load(f)

    categorical_cols = data.select_dtypes(include='object').columns

    one_hot_cols = ['MaturitySize',"Gender","FurLength","Vaccinated","Dewormed","Sterilized","Health"]


    for col in one_hot_cols:
        data[col] = data[col].astype('str')

    ohe_output = ohe.transform(data[one_hot_cols])
    ohe_output = pd.DataFrame(ohe_output)
    ohe_output.columns = ohe.get_feature_names_out(one_hot_cols)

    data = data.drop(one_hot_cols, axis=1)

    for col in label_encoders:
        data[col] = label_encoders[col].transform(data[col])

    # replacing categorical columns with encoded columns
    data = data.drop(columns=categorical_cols)
    data = pd.concat([data, ohe_output], axis=1)

    # scaling numerical columns
    numerical_cols = ['Age', 'Fee', 'Quantity']

    with open('C:/Users/Omen/Desktop/СС/PR5/pipeline/settings/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    scaled = scaler.transform(data[numerical_cols])
    data = data.drop(columns=numerical_cols)
    data = pd.concat([data, pd.DataFrame(scaled, columns=numerical_cols)], axis=1)

    return data