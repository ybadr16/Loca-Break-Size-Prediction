import pandas as pd
from sklearn.feature_selection import mutual_info_regression

# Load your data
path = './50%_processed.csv'
max_row = 300
df = pd.read_csv(path, header=0, nrows=max_row)

# Drop any non-numeric columns
numeric_df = df.select_dtypes(include=['number'])

# Compute mutual information between each feature and the dataset
mi_scores = []
for column in numeric_df.columns:
    mi_score = mutual_info_regression(numeric_df.drop(column, axis=1).dropna(), numeric_df[column].dropna())
    mi_scores.append(sum(mi_score))

# Create a DataFrame to store feature names and their mutual information scores
mi_scores_df = pd.DataFrame({'Feature': numeric_df.columns, 'Mutual_Information': mi_scores})

# Sort features by their mutual information scores in descending order
mi_scores_df = mi_scores_df.sort_values(by='Mutual_Information', ascending=False)


# Set display options to show all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
# Print the top features
print(mi_scores_df)
