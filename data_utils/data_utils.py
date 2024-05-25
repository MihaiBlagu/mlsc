import numpy as np
import os
from sklearn.model_selection import train_test_split

# Function to load data from .npz file
def load_data(file_path):
    data = np.load(file_path)
    th = data['th']  # theta values (output)
    u = data['u']    # input voltage values (input)
    return u, th


# Function to create input-output pairs
def create_IO_data(u, y, na, nb):
    X = []
    Y = []
    for k in range(max(na, nb), len(y)):
        X.append(np.concatenate([u[k-nb:k], y[k-na:k]]))
        Y.append(y[k])
    return np.array(X), np.array(Y)


# Function to split data into training, validation, and testing sets
def split_data(X, Y, val_size=0.2, test_size=0.1):
    # Split the data into training+validation and testing sets
    X_temp, X_test, Y_temp, Y_test = train_test_split(X, Y, test_size=test_size, shuffle=False)
    
    # Split the remaining data into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size=val_size / (1 - test_size), shuffle=False)
    
    return X_train, X_val, X_test, Y_train, Y_val, Y_test