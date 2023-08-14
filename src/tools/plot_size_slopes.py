import time
import os
import pandas as pd
import matplotlib.pyplot as plt

def main(path="E:\\Marcus\\gabby_test_data\\part11\\230427\\loc02"):
    file_path = os.path.join(path, "size", "slopes.csv")
    df = pd.read_csv(file_path, header=None)

    inc_values = df[df[0].str.startswith('inc')][1]
    dec_values = df[df[0].str.startswith('dec')][1].dropna()

    all_values = [inc_values, dec_values]
    labels = ['Increasing', 'Decreasing']

    plt.boxplot(all_values, labels=labels)

    plt.ylabel('Size Slopes')
    plt.title('Change in Capillary Size for Increasing and Decreasing Pressures')

    plt.show()

    

if __name__ == "__main__":
    ticks = time.time()
    main()
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))