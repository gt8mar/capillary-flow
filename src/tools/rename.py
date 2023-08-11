import os

# Set the directory path where the folders are located
directory = 'D:\\Marcus\\backup\\data\\part22\\230530'

# Loop through the range of numbers from 2 to 25
for i in range(2, 26):
    # Convert the number to a string and pad it with a leading zero if necessary
    old_folder_name = f'vid{i:02}'
    new_folder_name = f'vid{i-1:02}'

    # Construct the old and new folder paths
    old_path = os.path.join(directory, old_folder_name)
    new_path = os.path.join(directory, new_folder_name)

    # Rename the folder
    os.rename(old_path, new_path)
    print(f'Renamed {old_folder_name} to {new_folder_name}')