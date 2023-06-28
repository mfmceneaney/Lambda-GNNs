import argparse
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
  
#NOTE: CHECK OUT: https://www.geeksforgeeks.org/python-sqlite/

# sqliteConnection = None

# # Create parser and add arguments
# parser = argparse.ArgumentParser(description='Plot trial parameter distributions versus optimization metric from optuna SQL database')
# parser.add_argument('--dbname', type=str, default=None, help='path to sql database file')
# parser.add_argument('--xparam', type=str, default=None, help='x parameter name',default='Loss Coefficient $\\alpha$')
# parser.add_argument('--yparam', type=str, default=None, help='y parameter name',default='AUC')
# parser.add_argument('--logx', action="store_true", help='use log scale on x axis')
# parser.add_argument('--xlabel', type=str, default=None, help='x parameter label',default='alpha')
# parser.add_argument('--ylabel', type=str, default=None, help='y parameter label',default='AUC')
# parser.add_argument('--markeropt', type=str, default=None, help='marker option', default='bo')
# parser.add_argument('--markersize', type=int, default=None, help='marker option', default=1)
# args = parser.parse_args()

# # Set options
# dbName = args.dbname #'../drop/log_optimize_dagnn_8_9_22/log_optimize_dagnn_8_9_22.db'
# xparam = args.xparam #'alpha'
# yparam = args.yparam #'AUC' #NOTE: DEFAULT
# logx   = args.logx #False #NOTE: DEFAULT
# xlabel = args.xlabel #'Loss Coefficient $\\alpha$'
# ylabel = args.ylabel #'AUC' #NOTE: DEFAULT
# markeropt  = args.markeropt #'bo' #NOTE: DEFAULT
# markersize = args.markersize #1 #NOTE: DEFAULT

# Set Database name
parser = argparse.ArgumentParser(description='Plot trial parameter distributions versus optimization metric from optuna SQL database')
parser.add_argument('--dbname', type=str, default=None, help='path to sql database file')
args = parser.parse_args()
print("DEBUGGING: parser = ",parser)
dbname  = args.dbname
print("DEBUGGING: dbname = ",dbname)

# For GNNs
xparams = ['batch_size','lr','do','hdim','nlayers','nmlp']
xlabels = ['Batch size', 'Learning rate $\\eta$', 'Dropout rate', 'Hidden dimensions', 'Number of GIN layers', 'Number of MLP layers']
logxs   = [False,True,False,False,False,False]

# For DAGNNs
xparams = ['batch_size','alpha','lr', 'lr_c', 'lr_d','do','hdim','nlayers','nmlp','hdim_head','nmlp_head']
xlabels = ['Batch size', 'Loss coefficient $\\alpha$','Learning rate $\\eta$', 'Classifier learning rate $\\eta_C$', 'Discriminator learning rate $\\eta_D$', 'Dropout rate', 'Hidden dimensions', 'Number of GIN layers', 'Number of MLP layers','Head hidden dimensions','Number of head MLP layers']
logxs   = [False,False,True,True,True,False,False,False,False,False,False]

def myscript(
    dbName='',
    xparam = 'alpha',
    yparam = 'AUC',
    logx   = False,
    xlabel = 'Loss Coefficient $\\alpha$',
    ylabel = 'AUC',
    markeropt  = 'bo',
    markersize = 1
    ):
    """
    Description:
    """
    sqliteConnection = None
    try:

        #NOTE: BELOW IS FOR SCRIPT COMMENT OUT FOR FUNCTION WITH LIST
        # # Set options
        # dbName = args.dbname #'../drop/log_optimize_dagnn_8_9_22/log_optimize_dagnn_8_9_22.db'
        # xparam = args.xparam #'alpha'
        # yparam = args.yparam #'AUC' #NOTE: DEFAULT
        # logx   = args.logx #False #NOTE: DEFAULT
        # xlabel = args.xlabel #'Loss Coefficient $\\alpha$'
        # ylabel = args.ylabel #'AUC' #NOTE: DEFAULT
        # markeropt  = args.markeropt #'bo' #NOTE: DEFAULT
        # markersize = args.markersize #1 #NOTE: DEFAULT
        
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

#----- MAIN -----#
for i, el in enumerate(xparams):
    print("i, el = ",i,el)
    myscript(
        dbName     = dbname,
        xparam     = xparams[i],
        # yparam     = yparam,
        logx       = logxs[i],
        xlabel     = xlabels[i]#,
        # ylabel     = ylabel,
        # markeropt  = markeropt,
        # markersize = markersize
    )
    
print("DONE")
