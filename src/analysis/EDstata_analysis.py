import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def filter_dataframe(df, diag):
    # Check if diag is either 'HDDIAG' or 'DIAG' and set the appropriate column names
    if diag in ['HDDIAG', 'DIAG']:
        column_prefix = diag
    else:
        print(f"Invalid diag value: {diag}")
        return None

    # Generate column names to check
    column_names = [f'{column_prefix}{i}' for i in range(1, 6)]

    # Check each column and apply the filter if the column exists
    filters = []
    for col in column_names:
        if col in df.columns:
            filters.append(df[col].str.startswith('F'))

    # Combine filters with OR condition if any filters were added
    if filters:
        combined_filter = filters[0]
        for f in filters[1:]:
            combined_filter |= f

        behavorial_df = df[combined_filter]
        print(behavorial_df.shape)
        return behavorial_df
    else:
        print(f'None of the {column_prefix} columns are in the DataFrame')
        return None
    
def analyze_time_2(behavorial_df, category = 'LOV', plot = False):
    """
    Creates histogram and dataframe of ED visit time for patients with behavioral issues
    bins: [min(ed_time), 60, 120, 240, 360, 600, 840, 1440]

    Args:
        behavorial_df (pandas df): a dataframe with only entries that have 'F' in HDDIAG1, HDDIAG2, HDDIAG3, HDDIAG4, HDDIAG5
        category (str): 'LOV', 'LOS', 'BOARDED', 'OBSSTAY'
    
    Returns:
        edtime_df (pandas df): a dataframe with bin range, count, and percentage
    """
    # Put length of visit for these patients into different categories: [min(ed_time), 60, 120, 240, 360, 600, 840, 1440] and plot their percentages with percentage labels
    # wait times are in minutes
    ed_time = behavorial_df[category] #LOV, LOS, BOARDED, OBSSTAY
    if behavorial_df[category].isna().all():
        print(f'No {category} column')
        edtime_df = pd.DataFrame({
            'Bin Range': ['0 to 0'],
            'Count': [0],
            'Percentage': [0]
        })
        return edtime_df
    # print any ed times that are not a number:
    # print(ed_time[ed_time.isna()])
    total_patients = len(ed_time)
    if max(ed_time) > 1440:
        bins = [min(ed_time), 60, 120, 240, 360, 600, 840, 1440, max(ed_time)]
    else:
        bins = [min(ed_time), 60, 120, 240, 360, 600, 840, 1440]
    counts, bin_edges, patches = plt.hist(ed_time, bins = bins, edgecolor = 'black')
    percentages = 100 * counts / total_patients
    # plot the number in each bin on top of the bar
    for count, x in zip(percentages, bin_edges):
        plt.text(x, count, f'{count:.1f}%', va='center')
    plt.title(f'{category} time for patients with behavioral issues 2021')
    if plot:
        plt.show()
    else: 
        plt.close()
    bin_labels = [f'{bin_edges[i]} to {bin_edges[i+1]}' for i in range(len(bin_edges)-1)]

    # Create a DataFrame
    edtime_df = pd.DataFrame({
        'Bin Range': bin_labels,
        'Count': counts,
        'Percentage': percentages
    })
    return edtime_df

def main(path, plot = False, diag = 'HDDIAG', write = False):
    filename = os.path.basename(path).split('.')[0]
    filename = f'{filename}-behavorial-{diag}.xlsx'
    df = pd.read_stata(path, convert_categoricals=False)
    # print(df.head())

    # iterate through each column name and print it:
    for column in df.columns:
        if (column.startswith('BOARD') or column.startswith('LOV')) or column.startswith('WAIT') or column.startswith('BLANK'):
            print(column)
        # print(column)
    print(len(df.columns))

    # make new df with only entries that have 'F' in HDDIAG1, HDDIAG2, HDDIAG3, HDDIAG4, HDDIAG5
    behavorial_df = filter_dataframe(df, diag)
    if behavorial_df is None:
        return 1
    print(behavorial_df.head())
    
    # # if there are no patients with behavioral issues, return 1
    # if behavorial_df is None:
    #     return 1
    
    # # Put wait times for these patients into different categories: [<15 min, 15-59 min, 60-119 min, 120-179 min, 180-239 min, 240-359 min, >360 min] and plot their percentages with percentage labels
    # # wait times are in minutes
    # if 'WAITTIME' not in behavorial_df.columns or behavorial_df['WAITTIME'].isna().all():
    #     print('No WAITTIME column')
    # else:
    #     wait_times = behavorial_df['WAITTIME']
    #     print(wait_times)
    #     total_patients = len(wait_times)
    #     if max(wait_times) > 360:
    #         bins = [min(wait_times), 15, 60, 120, 180, 240, 360, max(wait_times)]
    #     else:
    #         bins = [min(wait_times), 15, 60, 120, 180, 240, 360]
    #     counts, bin_edges, patches = plt.hist(wait_times, bins = bins, edgecolor = 'black')
    #     percentages = 100 * counts / total_patients
    #     # plot the number in each bin on top of the bar
    #     for count, x in zip(percentages, bin_edges):
    #         plt.text(x, count, f'{count:.1f}%', va='center')
    #     plt.title('Wait times for patients with behavioral issues 2021')
    #     if plot:
    #         plt.show()
    #     else:
    #         plt.close()

    #     bin_labels = [f'{bin_edges[i]} to {bin_edges[i+1]}' for i in range(len(bin_edges)-1)]

    #     # Create a DataFrame
    #     wait_df = pd.DataFrame({
    #         'Bin Range': bin_labels,
    #         'Count': counts,
    #         'Percentage': percentages
    #     })
    # if 'WAITTIME' not in behavorial_df.columns or behavorial_df['WAITTIME'].isna().all(): 
    #     wait_df = pd.DataFrame({
    #         'Bin Range': ['0 to 0'],
    #         'Count': [0],
    #         'Percentage': [0]
    #     })
    # if 'LOV' in behavorial_df.columns:
    #     lov_df = analyze_time_2(behavorial_df, category = 'LOV', plot = False)
    # elif 'BLANK1' in behavorial_df.columns: 
    #     lov_df = analyze_time_2(behavorial_df, category = 'BLANK1', plot = False)
    # else:
    #     lov_df = pd.DataFrame({
    #         'Bin Range': ['0 to 0'],
    #         'Count': [0],
    #         'Percentage': [0]
    #     })
        
    # if 'BOARDED' in behavorial_df.columns:
    #     boarded_df = analyze_time_2(behavorial_df, category = 'BOARDED', plot = False)
    # else: 
    #     boarded_df = pd.DataFrame({
    #         'Bin Range': ['0 to 0'],
    #         'Count': [0],
    #         'Percentage': [0]
    #     })
    # if write:
    #     with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
    #         # Write each DataFrame to a different worksheet
    #         wait_df.to_excel(writer, sheet_name='WAITTIME')
    #         lov_df.to_excel(writer, sheet_name='LOV')
    #         boarded_df.to_excel(writer, sheet_name='BOARDED')

    #         # Close the Pandas Excel writer and output the Excel file
    #         # writer.save()
        
    return 0



    


if __name__ == '__main__':
    folder = 'C:\\Users\\gt8ma\\Downloads\\edstata'
    for file in os.listdir(folder):
        if file.endswith('.dta'):
            print(file)
            main(os.path.join(folder, file), diag = 'HDDIAG')
            main(os.path.join(folder, file), diag = 'DIAG')





    


