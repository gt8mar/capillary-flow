import time
import pandas as pd
import platform

def main():
    size_data_fp = '/hpc/projects/capillary-flow/results/size/size_data.csv'
    df = pd.read_csv(size_data_fp)
    # read csv file as a list of lists excluding header
    data = df.values.tolist()[1:] 
    # sort by participant, date, location, capnum, vidnum
    data.sort(key=lambda x: (x[0], x[1], x[2], x[5], x[6]))

    # create nested dictionary
    nested_data = {}

    for record in data:
        participant, date, location, capnum = record[0], record[1], record[2], record[5]
        
        if participant not in nested_data:
            nested_data[participant] = {}
        if date not in nested_data[participant]:
            nested_data[participant][date] = {}
        if location not in nested_data[participant][date]:
            nested_data[participant][date][location] = {}
        if capnum not in nested_data[participant][date][location]:
            nested_data[participant][date][location][capnum] = []
        
        nested_data[participant][date][location][capnum].append(record)


    # calculate compliance for each capillary, save to dictionary
    compliances = {}
    for participant in nested_data:
        compliances[participant] = []
        for date in nested_data[participant]:
            for location in nested_data[participant][date]:
                for capnum in nested_data[participant][date][location]:
                    capnum_entries = nested_data[participant][date][location][capnum]
                    capnum_entries.sort(key=lambda x: x[6])  # sort by vidnum
                    
                    area_initial = None
                    area_final = None
                    
                    # get the area_initial and area_final values at 0.2 and 1.2 psi
                    for entry in capnum_entries:
                        if entry[4] == 0.2:
                            area_initial = entry[2]  
                        elif entry[4] == 1.2:
                            area_final = entry[3] 
                            break  

                    # If both initial and final are found
                    if area_initial is not None and area_final is not None:
                        compliance = (area_final - area_initial) # divided by delta P = 1.2 - 0.2 = 1
                        compliances[participant].append(compliance)
                    else:
                        continue   
          
    # save to csv  
    df = pd.DataFrame.from_dict(compliances) 
    df.to_csv('/hpc/projects/capillary-flow/results/size/compliances.csv', index=False)         
    
                

"""
-----------------------------------------------------------------------------
"""
# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    ticks = time.time()
    main()
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))