import platform, os
import pandas as pd
import numpy as np
import seaborn as sns

def main(verbose = False):
    if platform.system() == 'Windows':
        if 'gt8mar' in os.getcwd():
            path = 'C:\\Users\\gt8mar\\capillary-flow\\results\\summary_df_test.csv'
        else:
            path = 'C:\\Users\\gt8ma\\capillary-flow\\results\\summary_df_test.csv'
    else:
        path = '/hpc/projects/capillary-flow/results/summary_df_test.csv'
    
    summary_df = pd.read_csv(path)
    print(summary_df.head())

if __name__ == '__main__':
    main()