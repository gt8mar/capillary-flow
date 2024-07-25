import os
import time


def main(path):
    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)

            index = filename.find('WkSl')
            
            if index != -1:
                # Check the character before 'WkSl'
                if index > 0:
                    char_before = filename[index - 1]
                    
                    if char_before == '_':
                        new_filename = filename
                    elif char_before == '-' or char_before == ' ':
                        new_filename = filename[:index - 1] + '_' + filename[index:]
                    else:
                        new_filename = filename[:index] + '_' + filename[index:]
                else:
                    # If 'WkSl' is at the beginning of the string
                    continue
                
                # Check if 'new_' exists in the filename
                new_index = new_filename.find('new_')
                if new_index != -1:
                    # Remove 'new_' from the new filename
                    new_filename = new_filename[:new_index] + new_filename[new_index + 4:]
                
            else:
                # If 'WkSl' is not found in the filename
                continue
            
            new_file_path = os.path.join(dirpath, new_filename)
            os.rename(file_path, new_file_path)
        

# TODO fix path
if __name__ == "__main__":
    ticks = time.time()
    path = 'J:\\frog\\data'
    main(path)
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))