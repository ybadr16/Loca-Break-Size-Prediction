import numpy as np
from sklearn.feature_selection import mutual_info_regression
import os
import pandas as pd
import numpy as np

sequences1 = list()
max_row = 300  # Change this to the desired maximum row number
path = './'
non_existing = ['./GPdata/17.0%_processed.csv', './GPdata/19.0%_processed.csv', './GPdata/22.0%_processed.csv', './GPdata/23.0%_processed.csv', './GPdata/24%_processed.csv']
for i in np.arange(0, 101, 0.5):
    if i % 1 != 0:
        i = i
        try:
            file_path = path + str(i) + '%_processed.csv'
            if file_path in non_existing:
                continue
            # Read only up to the specified maximum row
            df = pd.read_csv(file_path, header=0, nrows=max_row)
            # Drop column if it exists
            if 'hmi_RCSLT501_VALUE_BAR' in df.columns:
                df.drop(columns=['hmi_RCSLT501_VALUE_BAR'], inplace=True)
            if 'Variable' in df.columns:
                df.drop(columns=['Variable'], inplace=True)
            values = df.values
            sequences1.append(values)
        except FileNotFoundError:
            try:
                file_path = path + str(i) + '_processed.csv'
                # Read only up to the specified maximum row
                df = pd.read_csv(file_path, header=0, nrows=max_row)
                # Drop column if it exists
                if 'hmi_RCSLT501_VALUE_BAR' in df.columns:
                    df.drop(columns=['hmi_RCSLT501_VALUE_BAR'], inplace=True)
                if 'Variable' in df.columns:
                    df.drop(columns=['Variable'], inplace=True)
                values = df.values
                sequences1.append(values)
            except FileNotFoundError:
                print("File Not Found")
    else:
        i = int(i)
        try:
            file_path = path + str(i) + '.0%_processed.csv'
            if file_path in non_existing:
                continue
            # Read only up to the specified maximum row
            df = pd.read_csv(file_path, header=0, nrows=max_row)
            # Drop column if it exists
            if 'hmi_RCSLT501_VALUE_BAR' in df.columns:
                df.drop(columns=['hmi_RCSLT501_VALUE_BAR'], inplace=True)
            if 'Variable' in df.columns:
                df.drop(columns=['Variable'], inplace=True)
            values = df.values
            sequences1.append(values)
        except FileNotFoundError:
            try:
                file_path = path + str(i) + '%_processed.csv'
                # Read only up to the specified maximum row
                df = pd.read_csv(file_path, header=0, nrows=max_row)
                # Drop column if it exists
                if 'hmi_RCSLT501_VALUE_BAR' in df.columns:
                    df.drop(columns=['hmi_RCSLT501_VALUE_BAR'], inplace=True)
                if 'Variable' in df.columns:
                    df.drop(columns=['Variable'], inplace=True)
                values = df.values
                sequences1.append(values)
            except FileNotFoundError:
                print("File Not Found")


print(len(sequences1))

# Assume the data is loaded into a NumPy array called 'data'
data = np.array(sequences1) # Replace with your actual data

# Calculate mutual information across levels
mi_across_levels = []
for feature in range(data.shape[2]):
    feature_data = data[:, :, feature].flatten()
    mi_scores = mutual_info_regression(feature_data.reshape(-1, 1), np.arange(len(feature_data)))
    mi_across_levels.append(np.mean(mi_scores))

# Select features with the least mutual information across levels
num_features_across_levels = 10 # Specify the number of features to select
selected_features_across_levels = np.argsort(mi_across_levels)[:num_features_across_levels]

# Calculate mutual information within each level
mi_within_levels = []
for level in range(data.shape[0]):
    level_data = data[level]
    mi_matrix = np.zeros((data.shape[2], data.shape[2]))
    for i in range(data.shape[2]):
        for j in range(i+1, data.shape[2]):
            mi_score = mutual_info_regression(level_data[:, i].reshape(-1, 1), level_data[:, j])[0]
            mi_matrix[i, j] = mi_score
            mi_matrix[j, i] = mi_score
    mi_within_levels.append(mi_matrix)

# Select features with the highest mutual information within each level
num_features_within_level = 10  # Specify the number of features to select per level
selected_features_within_levels = []
for level in range(data.shape[0]):
    mi_matrix = mi_within_levels[level]
    selected_features = np.argsort(np.sum(mi_matrix, axis=1))[-num_features_within_level:]
    selected_features_within_levels.append(selected_features)

# Print the selected features
print("Selected features across levels:", selected_features_across_levels)
print("Selected features within each level:")
for level, features in enumerate(selected_features_within_levels):
    print(f"Level {level+1}: {features}")

def find_most_repeated_numbers(array_2d):
    # Flatten the 2D array and remove duplicates within each individual array
    flat_array = np.array([np.unique(sub_array) for sub_array in array_2d])

    # Flatten the 2D array into a single array
    flat_array = flat_array.flatten()

    # Count occurrences of each number
    unique_numbers, counts = np.unique(flat_array, return_counts=True)

    # Sort by counts in descending order
    sorted_indices = np.argsort(-counts)

    # Extract the 10 most repeating numbers
    most_repeated_numbers = unique_numbers[sorted_indices][:75]
    counts = counts[sorted_indices][:75]

    return most_repeated_numbers, counts


most_repeated_numbers, counts = find_most_repeated_numbers(selected_features_within_levels)
print("Most repeated numbers and their counts:")
for number, count in zip(most_repeated_numbers, counts):
    print("Number:", number, "Count:", count)
