import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
df = pd.read_csv('image_data.csv')

# Display DataFrame information to help you identify the numeric column
print("DataFrame information:")
print(df.info())

# Plot a histogram of a numeric column
# IMPORTANT: Replace 'severity' with the name of your numeric column
plt.hist(df['severity'], bins=10, edgecolor='black')
plt.title('Distribution of Severity Scores')
plt.xlabel('Severity')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()