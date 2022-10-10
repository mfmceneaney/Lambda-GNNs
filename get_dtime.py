import argparse
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
import sys
  
#NOTE: CHECK OUT: https://www.geeksforgeeks.org/python-sqlite/

sqliteConnection = None

try:

    # Set options
    dbName = sys.argv[1] #'log_optimize_dagnn_delta_fraction_0.2_9_6_22.db'
    xlabel = 'Total Time Taken (s)'
    ylabel = 'Counts' #NOTE: DEFAULT
    
    # Connect to DB and create a cursor
    sqliteConnection = sqlite3.connect(dbName)
    cursor = sqliteConnection.cursor()
    print('DB Init')
  
    # Write a query and execute it with cursor
    query = 'select sqlite_version();'
    query = 'select datetime_start, datetime_complete from trials where state like \'COMPLETE\';'
    print(query)#DEBUGGING
    cursor.execute(query)
  
    # Fetch results
    result = cursor.fetchall()

    # Convert to datetime objects
    datetime_start      = [datetime.strptime(el[0], '%Y-%m-%d %H:%M:%S.%f') for el in result]
    datetime_complete   = [datetime.strptime(el[1], '%Y-%m-%d %H:%M:%S.%f') for el in result]
    datetime_difference = [datetime_complete[i] - datetime_start[i] for i in range(len(datetime_start))]
    datetime_difference_seconds = [el.seconds for el in datetime_difference]
    average = np.average(datetime_difference_seconds)
    stddev  = np.std(datetime_difference_seconds)
    ave_dtime = timedelta(seconds=average)
    std_dtime = timedelta(seconds=stddev)

    # Print time differences
    print("Listing times:")
    for el in datetime_difference:
        print(str(el))
    print("Average = ",str(ave_dtime))
    print("Average = ",str(ave_dtime),"±",str(std_dtime))
    
    # Make plot of AUC (y axis) vs. param (x axis)
    f = plt.figure()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.hist(datetime_difference_seconds,bins=100,alpha=0.5)
    f.savefig('dtimes_10_10_22.png')#NOTE: Save figure to png file

    # Close the cursor
    cursor.close()
  
# Handle errors
except sqlite3.Error as error:
    print('Error occured - ', error)
  
# Close DB Connection irrespective of success
# or failure
finally:
    
    if sqliteConnection:
        sqliteConnection.close()
        print('SQLite Connection closed')
