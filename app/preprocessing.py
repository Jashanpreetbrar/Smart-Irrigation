import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

DATA_FILE = 'data/output.csv'
EXOG_FEATURES = ['Soil_pH', 'Temperature', 'Humidity', 'Wind_Speed', 'Soil_Quality']

def load_and_preprocess(target='N'):
    df = pd.read_csv(DATA_FILE, parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    df = df[df['Crop_Type'] == 'Wheat']
    df = pd.get_dummies(df, columns=['Soil_Type'], prefix='soil')
    
    all_features = EXOG_FEATURES + [col for col in df.columns if 'soil_' in col]
    df[target] = df[target].replace(0, np.nan)
    df[target].interpolate(method='time', inplace=True)
    
    return df, all_features

def scale_features(train, test, features):
    scaler = MinMaxScaler()
    train_scaled = train.copy()
    test_scaled = test.copy()
    
    train_scaled[features] = scaler.fit_transform(train[features])
    test_scaled[features] = scaler.transform(test[features])
    
    return train_scaled, test_scaled, scaler
