import os
import time
import csv

def main(path):
    output_folder = 'J:\\frog\\results\\all_data.csv'

    rbc_counts_folder = os.path.join(path, 'rbc_count')
    rbc_csv = os.path.join(rbc_counts_folder, 'edge_counts.csv')
    velocities_folder = os.path.join(path, 'velocities')
    velocities_csv = os.path.join(velocities_folder, [f for f in os.listdir(velocities_folder) if f.endswith('.csv')][0])

    # Read velocity data into memory
    with open(velocities_csv, mode='r', newline='', encoding='utf-8') as v_csvfile:
        v_csv_reader = csv.DictReader(v_csvfile)
        v_data = list(v_csv_reader)

    # Read RBC count data into memory
    with open(rbc_csv, mode='r', newline='', encoding='utf-8') as r_csvfile:
        r_csv_reader = csv.DictReader(r_csvfile)
        r_data = list(r_csv_reader)
        fieldnames = r_csv_reader.fieldnames

    # Open output CSV file for appending
    with open(output_folder, mode='a', newline='', encoding='utf-8') as output_csvfile:
        output_fieldnames = fieldnames + ['Velocity (um/s)']
        output_writer = csv.DictWriter(output_csvfile, fieldnames=output_fieldnames)
        
        if not os.path.isfile(output_folder):
            output_writer.writeheader()

        # Process each row in the RBC count data
        for r_row in r_data:
            r_date = r_row['Date']
            r_frog = r_row['Frog']
            r_side = r_row['Side']
            r_condition = r_row['Condition']
            r_capillary = r_row['Capillary']
            r_rbc_count = r_row['RBC Count']

            # Process each row in the velocity data
            for v_row in v_data:
                v_date = v_row['Date']
                v_frog = v_row['Frog']
                v_side = v_row['Side']
                v_condition = v_row['Condition']
                v_capillary = v_row['Capillary']
                v_velocity = v_row['Velocity (um/s)']

                # Check for matching rows and update
                if v_date == r_date and v_frog == r_frog and v_side == r_side and r_condition in v_condition and v_capillary == r_capillary:
                    r_row['Velocity (um/s)'] = v_velocity
                    output_writer.writerow(r_row)
                    break
                        

if __name__ == "__main__":
    ticks = time.time()
    umbrella_folder = 'J:\\frog\\data'
    for date in os.listdir(umbrella_folder):
        if not date.startswith('24'):
            continue
        if date == 'archive': 
            continue
        for frog in os.listdir(os.path.join(umbrella_folder, date)):
            if frog.startswith('STD'):
                continue
            if not frog.startswith('Frog'):
                continue   
            for side in os.listdir(os.path.join(umbrella_folder, date, frog)):
                if side.startswith('STD'):
                    continue
                if side == 'archive':
                    continue
                print('Processing: ' + date + ' ' + frog + ' ' + side)
                path = os.path.join(umbrella_folder, date, frog, side)
                main(path)
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))