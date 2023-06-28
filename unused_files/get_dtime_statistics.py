import argparse
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
import sys
  
#NOTE: CHECK OUT: https://www.geeksforgeeks.org/python-sqlite/

sqliteConnection = None

def get_statistics(dbName):
    # Set options
    # dbName = sys.argv[1] #'log_optimize_dagnn_delta_fraction_0.2_9_6_22.db'
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

    # # Print time differences
    # print("Listing times:")
    # for el in datetime_difference:
    #     print(str(el))
    # print("Average = ",str(ave_dtime))
    print("--------------------")
    print("DB name = ",dbName)
    print("Average = ",str(ave_dtime),"±",str(std_dtime))
    
    # # Make plot of AUC (y axis) vs. param (x axis)
    # f = plt.figure()
    # plt.xlabel(xlabel)
    # plt.ylabel(ylabel)
    # plt.hist(datetime_difference_seconds,bins=100,alpha=0.5)
    # f.savefig('dtimes_10_10_22.png')#NOTE: Save figure to png file

    # Close the cursor
    cursor.close()

    return ave_dtime, std_dtime

dbnames = [
    'log_optimize_dagnn_delta_fraction_0.8_9_6_22.db',
    'log_optimize_dagnn_delta_fraction_0.5_9_6_22.db',
    'log_optimize_dagnn_delta_fraction_0.2_9_6_22.db'
    # 'log_optimize_delta_fraction_0.8_9_6_22.db',
    # 'log_optimize_delta_fraction_0.5_9_6_22.db',
    # 'log_optimize_delta_fraction_0.2_9_6_22.db'
]

try:
    aves, stds = [], []
    for db in dbnames:
        ave, std = get_statistics(db)
        aves.append(ave)
        stds.append(std)

    new_aves = np.array([el.seconds for el in aves])
    new_stds = np.array([el.seconds for el in stds])

    n = len(new_stds)

    average = np.average(new_aves)
    stddev  = np.sqrt(np.sum(np.power(new_stds,2)))/ np.sqrt(n)

    ave = timedelta(seconds=average)
    std = timedelta(seconds=stddev)

    print("Overall statistics:")
    print("\tave ± sqrt(sum(square(stddev))) /sqrt(n) = ",ave,"±",std)
    print("\tn = ",n)

  
# Handle errors
except sqlite3.Error as error:
    print('Error occured - ', error)
  
# Close DB Connection irrespective of success
# or failure
finally:
    
    if sqliteConnection:
        sqliteConnection.close()
        print('SQLite Connection closed')
