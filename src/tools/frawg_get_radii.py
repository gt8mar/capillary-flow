import os
import time
import csv

PIX_UM = 0.8

def get_condition(filename):
    start_index = filename.find('WkSl')
    end_index = filename.find('Frog')
    start_index += len('WkSl')
    return filename[start_index:end_index].strip()

def main(path, date, frog, side):
    centerline_folder = os.path.join(path, 'centerlines', 'coords')
    output_file = 'J:\\frog\\results\\all_data.csv'

    if not os.path.exists(centerline_folder):
        print(f"Centerline folder {centerline_folder} does not exist.")
        return

    for centerline in os.listdir(centerline_folder):
        centerline_file = os.path.join(centerline_folder, centerline)
        
        if not centerline_file.endswith('.csv'):
            continue

        condition = get_condition(centerline)
        capillary = centerline[centerline.find('.csv') - 1]
        
        with open(centerline_file, 'r') as f:
            reader = csv.reader(f)
            rows = []
            for row in reader:
                if 'average' in row:
                    break
                if row and row[2].replace('.', '', 1).isdigit():
                    rows.append(row)

            if not rows:
                continue

            average_value_pix = sum(float(row[2]) for row in rows) / len(rows)
            average_value_um = average_value_pix / PIX_UM

        # Read existing data from output file
        existing_data = []
        if os.path.exists(output_file):
            with open(output_file, 'r', newline='', encoding='utf-8') as output_csvfile:
                reader = csv.DictReader(output_csvfile)
                existing_data = list(reader)

        # Check if row exists and update or append new row
        updated = False
        for row in existing_data:
            if (row['Date'] == date and row['Frog'] == frog and row['Side'] == side and 
                row['Condition'] == condition and row['Capillary'] == capillary):
                row['Average Thickness (um)'] = average_value_um
                updated = True
                break

        # Write updated data back to output file
        with open(output_file, 'w', newline='', encoding='utf-8') as output_csvfile:
            fieldnames = ['Date', 'Frog', 'Side', 'Condition', 'Capillary', 'RBC Count', 'Velocity (um/s)', 'Average Thickness (um)']
            writer = csv.DictWriter(output_csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(existing_data)
            

            


            

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
                main(path, date, frog, side)
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))