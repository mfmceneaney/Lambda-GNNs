import argparse
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
  
#NOTE: CHECK OUT: https://www.geeksforgeeks.org/python-sqlite/

sqliteConnection = None

try:

    # Set options
    dbName = '../drop/log_optimize_8_5_22.db'
    xparam = 'nlayers'
    yparam = 'AUC' #NOTE: DEFAULT
    logx = False #NOTE: DEFAULT
    xlabel = 'Number of Layers'
    ylabel = 'AUC' #NOTE: DEFAULT
    markeropt = 'bo' #NOTE: DEFAULT
    markersize = 1 #NOTE: DEFAULT
    
    # Connect to DB and create a cursor
    sqliteConnection = sqlite3.connect(dbName)
    cursor = sqliteConnection.cursor()
    print('DB Init')
  
    # Write a query and execute it with cursor
    query = 'select sqlite_version();'
    query = 'select trial_values.trial_id, trial_values.value, trial_params.param_value from trial_values join trial_params on trial_values.trial_id==trial_params.trial_id where trial_values.value<1.0 and param_name like \'{}\';'.format(xparam)
    print(query)#DEBUGGING
    cursor.execute(query)
  
    # Fetch results
    result = cursor.fetchall()

    # Get numpy array from result
    arr = np.array(result)
    print("DEBUGGING: np.shape(arr) = ",np.shape(arr))#DEBUGGING
    y = arr[:,1]
    y = 1-y
    x = arr[:,2]

    # Make plot of AUC (y axis) vs. param (x axis)
    f = plt.figure()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if logx:
        plt.semilogx(x, y, markeropt, markersize=markersize) #NOTE: bo stands for blue o marker
    else:
        plt.plot(x, y, markeropt, markersize=markersize) #NOTE: bo stands for blue o marker
    f.savefig('{}_vs_{}.png'.format(xparam,yparam))#NOTE: Save figure to png file

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
