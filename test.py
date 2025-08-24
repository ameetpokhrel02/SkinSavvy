# 
import numpy as np
import os
import pandas as pd

# Check if the file 'image_data.csv' exists in the current directory.
file_name = 'image_data.csv'
if os.path.exists(file_name):
    print(f"File '{file_name}' found. Reading the file...")
    # Read the CSV file into a pandas DataFrame.
    df = pd.read_csv(file_name)

    # Display the first 5 rows of the DataFrame to show the user the data.
    print("First 5 rows of the DataFrame:")
    print(df.head().to_markdown(index=False, numalign="left", stralign="left"))

    # Display information about the DataFrame, including column data types and non-null values.
    print("\nDataFrame information:")
    print(df.describe())
else:
    print(f"Error: The file '{file_name}' was not found in the current directory.")