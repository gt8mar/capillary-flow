import os
import write_background_file
import time

def main():
    data_folder = 'D:\\Marcus\\backup\\data'
    participant_set = {'part24'}

    for participant in os.listdir(data_folder):
        if participant in participant_set:
            for date in os.listdir(os.path.join(data_folder, participant)):
                if os.path.isdir(os.path.join(data_folder, participant, date)):
                    for location in os.listdir(os.path.join(data_folder, participant, date)):
                        if os.path.isdir(os.path.join(data_folder, participant, date, location)) and location == 'loc03':
                            for vid in os.listdir(os.path.join(data_folder, participant, date, location, 'vids')):
                                path = os.path.join(data_folder, participant, date, location, 'vids', vid)
                                write_background_file.main(path, 'mean', False)

if __name__ == "__main__":
    ticks = time.time()
    main()
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))