
import os
import scipy.io
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(mat_file_path):
    if not os.path.exists(mat_file_path):
        raise FileNotFoundError(f"File not found at {mat_file_path}")
    
    mat_data = scipy.io.loadmat(mat_file_path)
    features = mat_data['variablesMatrix'].T
    target = mat_data['coefStatMatrix'][0, :]
    
    columns = [f'Param_{i+1}' for i in range(features.shape[1])]
    df = pd.DataFrame(features, columns=columns)
    df['Resistance'] = target
    
    X = df.drop(columns=['Resistance'])
    y = df['Resistance']
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.125, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler
