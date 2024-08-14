import os
import pandas as pd

# List of relevant variables
relevant_variables = [
    "TSCWT1",
    "RCSCT5",
    "MRSPT514", "MRSPT524", "MRSPT534", "MRSPT544",
    "RCSLT501", "RCSLT502",
    "RCSFT414", "RCSFT424",
    "RCSLT470",
    "RCSPT405",
    "RCSTT413A", "RCSTT413B",
    "RCSTT423B", "RCSTT423C",
    "RCSTT433B",
    "RCSTT443B",
    "RCSTT453A",
    "RCSTT1734",
    "CTMTT60",
    "CTMPT1000A",
    "RRCTP10"
]

# Directory containing CSV files
input_directory = "input_files"

# Output directory for new CSV files
output_directory = "output_files"

# Ensure output directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Process each CSV file
for filename in os.listdir(input_directory):
    if filename.endswith(".csv"):
        input_filepath = os.path.join(input_directory, filename)
        output_filepath = os.path.join(output_directory, filename)

        # Read CSV file
        df = pd.read_csv(input_filepath)

        # Find columns with relevant headers (containing substrings)
        relevant_columns = [col for col in df.columns if any(var in col for var in relevant_variables)]

        # Save new CSV file with only relevant columns
        df[relevant_columns].to_csv(output_filepath, index=False)
