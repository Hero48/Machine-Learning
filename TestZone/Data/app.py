import pandas_datareader.data as pdr
import datetime
import os

start_date = datetime.datetime(1970, 1, 1)
end_date = datetime.datetime.now()

# Define a list of forex pairs to download data for
forex_pairs = ['USDJPY', 'EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF']

# Create a directory to store the downloaded data
directory = 'forex_data'

if not os.path.exists(directory):
    os.makedirs(directory)

# Download and save data for each forex pair
for pair in forex_pairs:
    # Download data
    retry_count=5
    data = pdr.get_data_fred('DEX' + pair, start_date, end_date, '15min')


    
    # Save data to CSV file
    filename = os.path.join(directory, pair + '.csv')
    data.to_csv(filename)
    
    print('Saved data for', pair, 'to', filename)
