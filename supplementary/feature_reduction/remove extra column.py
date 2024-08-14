import csv
import os

def remove_column_from_csv(file_path, column_name):
    # Temporary file to store modified data
    temp_file_path = file_path + '.tmp'

    with open(file_path, 'r', newline='') as csvfile, open(temp_file_path, 'w', newline='') as temp_csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = [field for field in reader.fieldnames if field != column_name]

        writer = csv.DictWriter(temp_csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            if column_name in row:
                del row[column_name]
            writer.writerow(row)

    # Replace the original file with the modified one
    os.remove(file_path)
    os.rename(temp_file_path, file_path)

def remove_column_from_all_csvs(directory, column_name):
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            remove_column_from_csv(file_path, column_name)

# Call the function to remove the column from all CSVs in the current directory
remove_column_from_all_csvs('.', 'hmi_RCSLT501_VALUE_BAR')
