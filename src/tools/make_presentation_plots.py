import matplotlib.pyplot as plt
import numpy as np
import time

def plot_area():
    x = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
    y = [5548.29, 4826.60, 5283.39, 5689.99, 4778.7, 3412.6]

    plt.plot(x, y)
    plt.scatter(x, y)
    plt.xlabel("Pressure (psi)")
    plt.ylabel("Area (um^2)")

    plt.ylim(0, max(y) + 1000)

    plt.show()


def plot_velocity():
    x = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
    y = [253.18, 204.37, 148.53, 121.29, 142.02, 1.01]

    plt.plot(x, y)
    plt.scatter(x, y)
    plt.xlabel("Pressure (psi)")
    plt.ylabel("Velocity (um/s)")

    plt.ylim(0, max(y) + 100)

    plt.show()


if __name__ == "__main__":
    ticks = time.time()
    plot_area()
    plot_velocity()
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))