import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import platform

def main(path="E:\\Marcus\\gabby_poster_data\\part09\\230414\\loc01"):
    participant = os.path.basename(os.path.dirname(os.path.dirname(path)))
    date = os.path.basename(os.path.dirname(path))
    location = os.path.basename(path)

    file_path = os.path.join(path, "size", "slopes.csv")
    df = pd.read_csv(file_path, header=None)

    inc_values = df[df[0].str.startswith('inc')][1].dropna()
    dec_values = df[df[0].str.startswith('dec')][1].dropna()

    all_values = [inc_values, dec_values]
    labels = ['Increasing', 'Decreasing']

    plt.boxplot(all_values, labels=labels)

    plt.ylabel('Size Slopes')
    plt.title('Change in Capillary Size for Increasing and Decreasing Pressures')

    filename = "set_01_" + participant + "_" + date + "_" + location + "_size_slopes.png"
    plot_fp = os.path.join(path, "size", "slopes")
    os.makedirs(plot_fp, exist_ok=True)
    plt.savefig(os.path.join(plot_fp, filename))

    if platform.system() != 'Windows':
        os.makedirs("/hpc/projects/capillary-flow/results/size/slopes", exist_ok=True)
        slope_boxplot_results_fp = os.path.join("/hpc/projects/capillary-flow/results/size/slopes" , "set_01_" + participant + "_" + date + "_" + location +  "_slopes_boxplot.png")
        plt.savefig(slope_boxplot_results_fp)

    

if __name__ == "__main__":
    ticks = time.time()
    main()
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))