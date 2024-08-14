import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import pandas as pd
import numpy as np

def load_data(path, max_row=300):
    sequences = list()
    for i in np.arange(0.5, 100.5, 0.5):
        if i % 1 != 0:
            file_path = path + f"{i}%_processed.csv"
            try:
                df = pd.read_csv(file_path, header=0, nrows=max_row, usecols=[0, 1, 2, 6, 8, 11, 13, 18, 19, 22])
                if 'hmi_RCSLT501_VALUE_BAR' in df.columns:
                    df.drop(columns=['hmi_RCSLT501_VALUE_BAR'], inplace=True)
                sequences.append(df.values)
            except FileNotFoundError:
                file_path = path + f"{i}_processed.csv"
                try:
                    df = pd.read_csv(file_path, header=0, nrows=max_row, usecols=[0, 1, 2, 6, 8, 11, 13, 18, 19, 22])
                    if 'hmi_RCSLT501_VALUE_BAR' in df.columns:
                        df.drop(columns=['hmi_RCSLT501_VALUE_BAR'], inplace=True)
                    sequences.append(df.values)
                except FileNotFoundError:
                    print(f"File not found: {file_path}")
        else:
            file_path = path + f"{int(i)}.0%_processed.csv"
            try:
                df = pd.read_csv(file_path, header=0, nrows=max_row, usecols=[0, 1, 2, 6, 8, 11, 13, 18, 19, 22])
                if 'hmi_RCSLT501_VALUE_BAR' in df.columns:
                    df.drop(columns=['hmi_RCSLT501_VALUE_BAR'], inplace=True)
                sequences.append(df.values)
            except FileNotFoundError:
                file_path = path + f"{int(i)}%_processed.csv"
                try:
                    df = pd.read_csv(file_path, header=0, nrows=max_row, usecols=[0, 1, 2, 6, 8, 11, 13, 18, 19, 22])
                    if 'hmi_RCSLT501_VALUE_BAR' in df.columns:
                        df.drop(columns=['hmi_RCSLT501_VALUE_BAR'], inplace=True)
                    sequences.append(df.values)
                except FileNotFoundError:
                    print(f"File not found: {file_path}")
    return sequences

def preprocess(X, means, stds):
    return (X - means) / stds

def preprocess_output(y, mean, std):
    return (y - mean) / std

def postprocess_output(y, mean, std):
    return (y * std) + mean
