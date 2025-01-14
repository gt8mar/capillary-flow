# Re-import necessary libraries and re-define the plot code due to environment reset
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.font_manager import FontProperties

def main():
    # Set up style and font
    sns.set_style("whitegrid")
    source_sans = FontProperties(fname='C:\\Users\\gt8ma\\Downloads\\Source_Sans_3\\static\\SourceSans3-Regular.ttf')
    
    plt.rcParams.update({
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'font.size': 7, 'axes.labelsize': 7,
        'xtick.labelsize': 6, 'ytick.labelsize': 6,
        'legend.fontsize': 5, 'lines.linewidth': 0.5
    })

    # Parameters
    start_pressure = 0.2  # psi
    end_pressure = 1.2    # psi
    step_pressure = 0.2   # psi
    hold_time = 5         # seconds
    step_time = 10.49         # seconds

    # Generate pressure values
    pressure_increase = np.arange(start_pressure, end_pressure + step_pressure, step_pressure)
    pressure_decrease = np.arange(end_pressure, start_pressure - step_pressure, -step_pressure)

    # Adjust times to reflect step changes
    times_step = []
    pressures_step = []

    # Generate step-increase phase
    for i, pressure in enumerate(pressure_increase):
        times_step.extend([i * step_time, (i + 1) * step_time])
        pressures_step.extend([pressure, pressure])

    # # Add hold at peak pressure
    # times_step.extend([times_step[-1], times_step[-1] + hold_time])
    # pressures_step.extend([end_pressure, end_pressure])

    # Generate step-decrease phase
    for i, pressure in enumerate(pressure_decrease[1:], start=1):  # Skip the first as it's already included
        times_step.extend([times_step[-1], times_step[-1] + step_time])
        pressures_step.extend([pressure, pressure])

    # # Add hold at base pressure
    # times_step.extend([times_step[-1], times_step[-1] + hold_time])
    # pressures_step.extend([start_pressure, start_pressure])

    # Plot
    plt.figure(figsize=(2.4, 2.0))
    plt.plot(times_step, pressures_step, marker='o', markersize=2, color='tab:blue')
    plt.title('Pressure vs Time', fontproperties=source_sans)
    plt.xlabel('Time (s)', fontproperties=source_sans)
    plt.ylim(0, 1.4)
    plt.ylabel('Pressure (psi)', fontproperties=source_sans)
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig('C:\\Users\\gt8ma\\capillary-flow\\pressure_example.png', dpi = 400)
    plt.close()

if __name__ == '__main__':
    main()
