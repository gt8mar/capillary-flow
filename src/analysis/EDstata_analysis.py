import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_stata('C:\\Users\\gt8ma\\Downloads\\ed2021-stata\\ed2021-stata.dta', convert_categoricals=False)
print(df.head())

# iterate through each column name and print it:
for column in df.columns:
    if (column.startswith('HDDIAG') or column.startswith('DIAG')):
        print(column)
    # print(column)
    pass
print(len(df.columns))

# # print all unique entries in 'VMONTH'
# print(df['VMONTH'].unique())

diag1s = df['DIAG1'].unique()
# select DIAG1s that start with 'F'
f_diag1s = [diag1 for diag1 in diag1s if diag1.startswith('F')]
print(f_diag1s)
print(len(f_diag1s))

boards = df['BOARD'].unique()
print(boards)

# make new df with only entries that have 'F' in HDDIAG1, HDDIAG2, HDDIAG3, HDDIAG4, HDDIAG5
behavorial_df = df[df['HDDIAG1'].str.startswith('F') | df['HDDIAG2'].str.startswith('F') | df['HDDIAG3'].str.startswith('F') | df['HDDIAG4'].str.startswith('F') | df['HDDIAG5'].str.startswith('F')]
print(behavorial_df.shape)

# Put wait times for these patients into different categories: [<15 min, 15-59 min, 60-119 min, 120-179 min, 180-239 min, 240-359 min, >360 min] and plot their percentages with percentage labels
# wait times are in minutes
wait_times = behavorial_df['WAITTIME']
# print any wait times that are not a number:
# print(wait_times[wait_times.isna()])

total_patients = len(wait_times)
bins = [min(wait_times), 15, 60, 120, 180, 240, 360]
counts, bin_edges, patches = plt.hist(wait_times, bins = bins, edgecolor = 'black')
percentages = 100 * counts / total_patients
# plot the number in each bin on top of the bar
for count, x in zip(percentages, bin_edges):
    plt.text(x, count, f'{count:.1f}%', va='center')
plt.title('Wait times for patients with behavioral issues 2021')
plt.show()

bin_labels = [f'{bin_edges[i]} to {bin_edges[i+1]}' for i in range(len(bin_edges)-1)]

# Create a DataFrame
wait_df = pd.DataFrame({
    'Bin Range': bin_labels,
    'Count': counts,
    'Percentage': percentages
})

print(wait_df)






# Put length of visit for these patients into different categories: [<15 min, 15-59 min, 60-119 min, 120-179 min, 180-239 min, 240-359 min, >360 min] and plot their percentages with percentage labels
# wait times are in minutes
ed_time = behavorial_df['LOV']
# print any ed times that are not a number:
# print(ed_time[ed_time.isna()])
total_patients = len(ed_time)
bins = [min(wait_times), 60, 120, 240, 360, 600, 840, 1440]
counts, bin_edges, patches = plt.hist(wait_times, bins = bins, edgecolor = 'black')
percentages = 100 * counts / total_patients
# plot the number in each bin on top of the bar
for count, x in zip(percentages, bin_edges):
    plt.text(x, count, f'{count:.1f}%', va='center')
plt.title('ED visit time for patients with behavioral issues 2021')
plt.show()

bin_labels = [f'{bin_edges[i]} to {bin_edges[i+1]}' for i in range(len(bin_edges)-1)]

# Create a DataFrame
edtime_df = pd.DataFrame({
    'Bin Range': bin_labels,
    'Count': counts,
    'Percentage': percentages
})

print(edtime_df)






# plot histogram of wait times and 'BOARD' for all entries
# fig, ax = plt.subplots(1, 2, figsize=(15, 5))
# ax[0].hist(df['WAITTIME'], bins=10)
# ax[0].set_title('Wait times for all entries')
# ax[0].set_xlabel('Wait time (minutes)')
# ax[0].set_ylabel('Count')
# ax[1].hist(df['BOARD'], bins=10)
# ax[1].set_title('Board for all entries')
# ax[1].set_xlabel('Board')
# ax[1].set_ylabel('Count')
# plt.show()



# # plot histogram of wait times per month for each month
# fig, ax = plt.subplots(4, 3, figsize=(15, 15))
# for i in range(1, 13):
#     ax[(i-1)//3, (i-1)%3].hist(df[df['VMONTH'] == i]['WAITTIME'], bins=10)
#     ax[(i-1)//3, (i-1)%3].set_title(f'Wait times for month {i}')
#     ax[(i-1)//3, (i-1)%3].set_xlabel('Wait time (minutes)')
#     ax[(i-1)//3, (i-1)%3].set_ylabel('Count')
# plt.show()


