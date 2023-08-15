"""
Filename: count_seg_moco.py
---------------------------
This script counts the number of segmentation and motion correction files 
in each location directory.

By: Marcus Forst
"""

import os
import pandas as pd
from find_earliest_date_dir import find_earliest_date_dir



def count_seg_moco(dir_path):
    num_seg = 0
    num_moco = 0



    # for item in os.listdir(dir_path):
    #     item_path = os.path.join(dir_path, item)
        
    #     if os.path.isdir(item_path):
    #         if item.startswith('loc'):
    #             for sub_item in os.listdir(item_path):
    #                 sub_item_path = os.path.join(item_path, sub_item)
    #                 if sub_item.endswith('seg.png') and os.path.isfile(sub_item_path):
    #                     num_seg += 1
    #                 elif sub_item == 'moco' and os.path.isdir(sub_item_path):
    #                     num_moco += 1

    #         count_seg_moco(item_path, data)

    # if num_seg > 0 or num_moco > 0:
    #     data['location'].append(dir_path)
    #     data['num_seg'].append(num_seg)
    #     data['num_moco'].append(num_moco)

def main():
    root_directory = '/hpc/projects/capillary-flow/data'
    for i in range(9, 21):
        participant = 'part' + str(i).zfill(2)
        date = find_earliest_date_dir(os.path.join(root_directory, participant))
        df = pd.DataFrame(columns=['location', 'num_vids', 'num_seg'])
        for location in os.listdir(os.path.join(root_directory, participant, date)):
            print(f'participant: {participant}, date: {date}, location: {location}')
            vids = os.listdir(os.path.join(root_directory, participant, date, location, 'vids'))
            if len(vids) == 0:
                print(f'participant: {participant}, date: {date}, location: {location} has no videos')
            num_vids = len(vids)
            segs = os.listdir(os.path.join(root_directory, participant, date, location, 'segmented'))
            num_seg = len([seg for seg in segs if seg.endswith('seg.png')])
            row = {'location': location, 'num_vids': num_vids, 'num_seg': num_seg}
            df = pd.concat([df, pd.DataFrame(row, index=[0])])
        print(df)
            
            

            # data = count_seg_moco(dir_path)
            # df = pd.DataFrame(data)
            # print(df)

if __name__ == '__main__':
    main()

        

