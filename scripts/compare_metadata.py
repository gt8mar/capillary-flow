import os
import pandas as pd

def compare_excel_files(file1, file2):
    df1 = pd.read_excel(file1)
    df2 = pd.read_excel(file2)

    if df1.equals(df2):
        return "Files are identical"
    else:
        differences = df1.compare(df2)
        return differences

def compare_folders(folder1, folder2):
    files1 = set(os.listdir(folder1))
    files2 = set(os.listdir(folder2))

    common_files = files1.intersection(files2)
    unique_to_folder1 = files1 - files2
    unique_to_folder2 = files2 - files1

    report = {"identical": [], "different": [], "unique_to_folder1": list(unique_to_folder1), "unique_to_folder2": list(unique_to_folder2)}

    for file in common_files:
        file1 = os.path.join(folder1, file)
        file2 = os.path.join(folder2, file)
        result = compare_excel_files(file1, file2)
        if isinstance(result, str):
            report["identical"].append(file)
        else:
            report["different"].append({file: result})

    return report

folder1 = "C:\\Users\\gt8mar\\capillary-flow\\metadata"
folder2 = "C:\\Users\\gt8mar\\capillary-flow\\tests\\metadata_ejer"

comparison_report = compare_folders(folder1, folder2)
print(comparison_report)
