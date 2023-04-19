import argparse
import sqlite3
import numpy as np
  
#NOTE: CHECK OUT: https://www.geeksforgeeks.org/python-sqlite/

sqliteConnection = None

try:

    # Set options
    dbName = '../drop/log_optimize_dagnn_8_9_22/log_optimize_dagnn_8_9_22.db'
    xparam = 'alpha'
    value_max = 0.1089 #NOTE: DEFAULT
    ndigits_w = 0 #NOTE: DEFAULT
    ndigits_x = 1 #NOTE: DEFAULT
    ndigits_y = 5 #NOTE: DEFAULT
    
    # Connect to DB and create a cursor
    sqliteConnection = sqlite3.connect(dbName)
    cursor = sqliteConnection.cursor()
    print('DB Init')
  
    # Write a query and execute it with cursor
    query = 'select sqlite_version();'
    query = 'select trial_values.trial_id, trial_values.value, trial_params.param_value from trial_values join trial_params on trial_values.trial_id==trial_params.trial_id where trial_values.value<1.0 and param_name like \'{}\';'.format(xparam)
    query = 'select trial_values.trial_id, trial_values.value, trial_params.param_value from trial_values join trial_params on trial_values.trial_id==trial_params.trial_id where trial_values.value<{} and param_name like \'{}\' ORDER BY trial_values.value;'.format(value_max,xparam)

    print(query)#DEBUGGING
    cursor.execute(query)
  
    # Fetch results
    result = cursor.fetchall()

    # Get numpy array from result
    arr = np.array(result)
    print("DEBUGGING: np.shape(arr) = ",np.shape(arr))#DEBUGGING
    w = arr[:,0]
    y = arr[:,1]
    y = 1-y
    x = arr[:,2]

    # Print latex tables
    print("trial_id")
    print(' & '.join([str(round(el,ndigits_w)) if ndigits_w>0 else str(int(el)) for el in w]),'\\\\')
    print("trial_params.param_value : {}".format(xparam))
    print(' & '.join([str(round(el,ndigits_x)) if ndigits_x>0 else str(int(el)) for el in x]),'\\\\')
    print("trial_values.value")
    print(' & '.join([str(round(el,ndigits_y)) if ndigits_y>0 else str(int(el)) for el in y]),'\\\\')

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
