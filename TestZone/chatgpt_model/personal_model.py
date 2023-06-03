import pandas as pd

# Load the CSV file into a Pandas DataFrame
df = pd.read_csv('GBPCAD5m.csv')

# Drop any rows with missing data
df = df.dropna()

# Convert the 'Date' column to a datetime object
df['Date'] = pd.to_datetime(df['Date'])

# Set the 'Date' column as the index
df = df.set_index('Date')

print(df.shape)
