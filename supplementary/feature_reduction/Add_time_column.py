import pandas as pd
import os

# Get a list of all CSV files in the current directory
csv_files = [file for file in os.listdir() if file.endswith('.csv')]

# Iterate through each CSV file
for file in csv_files:
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file)

    # Ensure DataFrame has at most 310 rows
    if len(df) < 310:
        df = df.iloc[:310]

    # Create a new column named "time" with values from 1 to 310
    df['time'] = range(1, len(df) + 1)

    # Save the DataFrame back to the CSV file
    df.to_csv(file, index=False)
