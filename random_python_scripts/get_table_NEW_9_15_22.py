import argparse
import sqlite3
import numpy as np
  
#NOTE: CHECK OUT: https://www.geeksforgeeks.org/python-sqlite/

sqliteConnection = None

parser = argparse.ArgumentParser(description='Get parameter tables in LaTeX format from SQLITE3 optuna DB')
parser.add_argument('--dbname', type=str, default="test.db",
                    help='Path to sqlite3 database file') #NOTE: Needs to be in ~/.dgl
parser.add_argument('--xparam', type=str, default='alpha',
                    help='X parameter for table')
parser.add_argument('--max', type=float, default=1.0,
                    help='Maximum metric value on which to filter')
parser.add_argument('--ndigits_w', type=int, default=0,
                    help='Precision of table entries for trial_id')
parser.add_argument('--ndigits_x', type=int, default=1,
                    help='Precision of table entries for xparam')
parser.add_argument('--ndigits_y', type=int, default=5,
                    help='Precision of table entries for trial_values.value')

args = parser.parse_args()

try:

    # Set options
    dbName = args.dbname
    xparam = args.xparam
    value_max = args.max #NOTE: DEFAULT 0.1089
    ndigits_w = args.ndigits_w #NOTE: DEFAULT 0 
    ndigits_x = args.ndigits_x #NOTE: DEFAULT 1
    ndigits_y = args.ndigits_y #NOTE: DEFAULT 5
    
    # Connect to DB and create a cursor
    sqliteConnection = sqlite3.connect(dbName)
    cursor = sqliteConnection.cursor()
    print('DB Init')
  
    # Write a query and execute it with cursor
    query = 'select sqlite_version();'
    query = 'select trial_values.trial_id, trial_values.value, trial_params.param_value from trial_values join trial_params on trial_values.trial_id==trial_params.trial_id where trial_values.value<1.0 and param_name like \'{}\';'.format(xparam)
    query = 'SELECT trial_values.trial_id, trial_values.value, trial_params.param_value FROM trial_values JOIN trial_params ON trial_values.trial_id==trial_params.trial_id WHERE trial_values.value<{} AND trial_params.param_name LIKE \'{}\' ORDER BY trial_values.value;'.format(value_max,xparam)

    print(query)#DEBUGGING
    cursor.execute(query)
  
    # Fetch results
    result = cursor.fetchall()

    # Get numpy array from result
    arr = np.array(result)
    print("DEBUGGING: np.shape(arr) = ",np.shape(arr))#DEBUGGING
    w = arr[:,0]
    y = arr[:,1]
    print("DEBUGGING: y = ",y)#DEBUGGING
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
