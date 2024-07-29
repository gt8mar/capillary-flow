import os, platform
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.cm import ScalarMappable
import seaborn as sns
from scipy.integrate import simps, trapezoid
from scipy.stats import skew, kurtosis, wilcoxon, mannwhitneyu, kstest, ks_2samp, ks_1samp, wasserstein_distance
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from src.tools.parse_filename import parse_filename
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression, LinearRegression, LassoCV
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, auc, confusion_matrix, roc_curve, recall_score, precision_score, f1_score, r2_score, mean_squared_error, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
from matplotlib.colors import to_rgb, LinearSegmentedColormap
import colorsys
import matplotlib.patches as mpatches



# #For editable text. Except latex text is still shapes sadly
# mpl.rcParams['pdf.fonttype'] = 42
# mpl.rcParams['ps.fonttype'] = 42
# sns.set_style("whitegrid")
# font = {'size' : 6}
# lines = {'linewidth' : 0.5}
# fig = {'figsize' : (2.5, 1.5)}
# mpl.rc('font', **font)
# mpl.rc('lines', **lines)
# mpl.rc('figure', **fig)
# #Set style
# sns.set_theme(style="whitegrid", palette="pastel", color_codes=True)


def to_rgb(hex_color):
    """Converts a hex color to an RGB tuple."""
    hex_color = hex_color.lstrip('#')
    lv = len(hex_color)
    return tuple(int(hex_color[i:i + lv // 3], 16) / 255.0 for i in range(0, lv, lv // 3))

def create_monochromatic_palette(base_color, n_colors=5):
    """Creates a monochromatic palette based on the given color."""
    rgb = to_rgb(base_color)
    h, l, s = colorsys.rgb_to_hls(*rgb)
    
    colors = []
    # Increasing the spread for more distinct colors
    lightness_increment = 0.4 / (n_colors - 1)  # Adjust the 0.4 value to increase or decrease contrast
    for i in range(n_colors):
        l_new = max(0, min(1, l + (i - n_colors / 2) * lightness_increment))
        rgb_new = colorsys.hls_to_rgb(h, l_new, s)
        colors.append(rgb_new)
    # plot all the colors in the set:
    # for color in colors:
    #     plt.axhspan(0, 1, color=color)
    #     plt.show()
    return colors

def adjust_saturation_of_colors(color_list, saturation_scale=10):
    """Adjusts the saturation of a list of RGB colors."""
    adjusted_colors = []
    for color in color_list:
        h, l, s = colorsys.rgb_to_hls(*color)
        s_new = max(0, min(1, s + saturation_scale))
        rgb_new = colorsys.hls_to_rgb(h, l, s_new)
        adjusted_colors.append(rgb_new)
    return adjusted_colors

def adjust_brightness_of_colors(color_list, brightness_scale=0.1):
    """Adjusts the brightness (lightness) of a list of RGB colors."""
    adjusted_colors = []
    for color in color_list:
        h, l, s = colorsys.rgb_to_hls(*color)
        l_new = max(0, min(1, l + brightness_scale))
        rgb_new = colorsys.hls_to_rgb(h, l_new, s)
        adjusted_colors.append(rgb_new)
    return adjusted_colors


# Function to calculate mean, standard error, and 95% CI
def calculate_stats(group, dimensionless = False):
    if dimensionless:
        mean = group['Dimensionless Velocity'].mean()
        sem = stats.sem(group['Dimensionless Velocity'])
        ci = 1.96 * sem
        return pd.Series({'Mean Dimensionless Velocity': mean, 'Lower Bound': mean - ci, 'Upper Bound': mean + ci})
    else:
        mean = group['Corrected Velocity'].mean()
        sem = stats.sem(group['Corrected Velocity'])
        ci = 1.96 * sem
        return pd.Series({'Mean Velocity': mean, 'Lower Bound': mean - ci, 'Upper Bound': mean + ci})

def plot_CI(df, variable='Age', method='bootstrap', n_iterations=1000, 
            ci_percentile=99.5, write = True, dimensionless=False, video_median = False):
    """Plots the mean/median and CI for the variable of interest.

    This function creates a plot comparing different groups based on the specified
    variable, showing either mean or median values with confidence intervals.

    Args:
        df (pd.DataFrame): The dataframe containing the data to be plotted.
        variable (str, optional): The variable to group by. Options are 'Age', 'SYS_BP', or 'Sex'. Defaults to 'Age'.
        method (str, optional): The method for calculating CI. Options are 'bootstrap' or 't-distribution'. Defaults to 'bootstrap'.
        n_iterations (int, optional): The number of iterations for bootstrap method. Defaults to 1000.
        ci_percentile (float, optional): The percentile for the confidence interval. Defaults to 99.5.
        write (bool, optional): Whether to write the plot to a file. Defaults to True.
        dimensionless (bool, optional): Whether to plot dimensionless velocity. Defaults to False.
        video_median (bool, optional): Whether to use video medians. Defaults to False.

    Returns:
        int: 0 if the plot was created successfully.

    Raises:
        ValueError: If an unsupported variable is specified.

    Note:
        This function assumes the existence of helper functions `calculate_median_ci` and `calculate_stats`.
        It also requires the Source Sans 3 font to be installed and accessible.
    """
    # Set up style and font
    sns.set_style("whitegrid")
    source_sans = FontProperties(fname='C:\\Users\\gt8ma\\Downloads\\Source_Sans_3\\static\\SourceSans3-Regular.ttf')
    
    plt.rcParams.update({
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'font.size': 7, 'axes.labelsize': 7,
        'xtick.labelsize': 6, 'ytick.labelsize': 6,
        'legend.fontsize': 5, 'lines.linewidth': 0.5
    })

    # Set color palette based on variable
    if variable == 'Age':
        base_color = '#1f77b4'##3FCA54''BDE4A7 #A1E5AB
    elif variable == 'SYS_BP':
        base_color = '2ca02c'#80C6C3 #ff7f0e
    elif variable == 'Sex':
        base_color = '674F92'#947EB0#2ca02c#CAC0D89467bd
    else:
        raise ValueError(f"Unsupported variable: {variable}")

    palette = create_monochromatic_palette(base_color)
    # palette = adjust_saturation_of_colors(palette, saturation_scale=1.3)
    palette = adjust_brightness_of_colors(palette, brightness_scale=.2)
    sns.set_palette(palette)
    if video_median:
        # collapse the data for each participant and video to a single row by selecting just the first row for each participant and video combination
        df = df.groupby(['Participant', 'Video', 'Capillary']).first().reset_index()
        # rename the 'Corrected Velocity' column to 'Velocity' for consistency
        df.rename(columns={'Dimensionless Velocity': 'Dimensionless Velocity OG'}, inplace=True) 
        # rename Median Dimensionless Velocity to Dimensionless Velocity
        df.rename(columns={'Video Median Dimensionless Velocity': 'Dimensionless Velocity'}, inplace=True)      
    # Group data
    group_col = f'{variable} Group'
    df[group_col] = np.select([
        df[variable] <= 50 if variable == 'Age' else df[variable] < 120 if variable == 'SYS_BP' else df[variable] == 'M',
        df[variable] > 50 if variable == 'Age' else df[variable] >= 120 if variable == 'SYS_BP' else df[variable] == 'F'
    ], ['≤50' if variable == 'Age' else '<120' if variable == 'SYS_BP' else 'M',
        '>50' if variable == 'Age' else '≥120' if variable == 'SYS_BP' else 'F'])

    # Calculate stats
    stats_func = calculate_median_ci if method == 'bootstrap' else calculate_stats
    stats_df = df.groupby([group_col, 'Pressure']).apply(stats_func, ci_percentile=ci_percentile, dimensionless=dimensionless).reset_index()

    # Plot
    fig, ax = plt.subplots(figsize=(2.4, 2.0))

    for i, (label, group_df) in enumerate(stats_df.groupby(group_col)):
        if i == 0:
            i_color = 0
            dot_color = 0
        elif i == 1:
            i_color = 3
            dot_color = 2
        elif i == 2:
            i_color = 4
            dot_color=3
        if dimensionless:
            y_col = 'Median Dimensionless Velocity' if method == 'bootstrap' else 'Mean Dimensionless Velocity'
        else:
            y_col = 'Median Velocity' if method == 'bootstrap' else 'Mean Velocity'
        lower_col = 'CI Lower Bound' if method == 'bootstrap' else 'Lower Bound'
        upper_col = 'CI Upper Bound' if method == 'bootstrap' else 'Upper Bound'
        
        ax.errorbar(group_df['Pressure'], group_df[y_col], 
                    yerr=[group_df[y_col] - group_df[lower_col], group_df[upper_col] - group_df[y_col]],
                    label=f'{variable} Group {label}', fmt='-o',markersize=2, color=palette[dot_color])
        ax.fill_between(group_df['Pressure'], group_df[lower_col], group_df[upper_col], alpha=0.4, color=palette[i_color])

    legend_handles = [mpatches.Patch(color=palette[0], label=f'{variable} Group ≤50' if variable == 'Age' else '<120' if variable == 'SYS_BP' else 'M', alpha=0.6),
                      mpatches.Patch(color=palette[3], label=f'{variable} Group >50' if variable == 'Age' else '≥120' if variable == 'SYS_BP' else 'F', alpha=0.6)]

    ax.set_xlabel('Pressure (psi)', fontproperties=source_sans)
    if dimensionless:
        ax.set_ylabel('Dimensionless Velocity', fontproperties=source_sans)
        ax.set_title(f'{"Median" if method == "bootstrap" else "Mean"} Dimensionless Velocity vs. Pressure with {ci_percentile}% CI', fontproperties=source_sans, fontsize=8)
    else:
        ax.set_ylabel('Velocity (um/s)', fontproperties=source_sans)
        ax.set_title(f'{"Median" if method == "bootstrap" else "Mean"} Velocity vs. Pressure with {ci_percentile}% CI', fontproperties=source_sans, fontsize=8)
    ax.legend(handles = legend_handles, prop=source_sans)
    ax.grid(True, linewidth=0.3)

    plt.tight_layout()
    if write:
        if video_median:
            plt.savefig(f'C:\\Users\\gt8ma\\capillary-flow\\results\\{variable}_videomedians_CI.png', dpi = 600)
        else:
            plt.savefig(f'C:\\Users\\gt8ma\\capillary-flow\\results\\{variable}_CI.png', dpi = 600)
    else:
        plt.show()
    return 0

def calculate_overlap(ci1, ci2):
    lower1, upper1 = ci1
    lower2, upper2 = ci2
    intersection_lower = max(lower1, lower2)
    intersection_upper = min(upper1, upper2)
    if intersection_upper > intersection_lower:
        intersection_length = intersection_upper - intersection_lower
        smaller_interval_length = min(upper1 - lower1, upper2 - lower2)
        return intersection_length / smaller_interval_length
    return 0

def plot_CI_overlaps(df, variable='Age', method='bootstrap', n_iterations=1000, ci_percentile=99.5):
    # Define groups based on variable
    if variable == 'Age':
        df['Group'] = np.where(df['Age'] <= 50, '≤50', '>50')
    elif variable == 'SYS_BP':
        df['Group'] = np.where(df['SYS_BP'] < 120, '<120', '≥120')
    elif variable == 'Sex':
        df['Group'] = np.where(df['Sex'] == 'M', 'M', 'F')
    else:
        raise ValueError('Variable not recognized')
    
    group_cis = {}
    
    # Bootstrap method for confidence intervals
    for group, group_df in df.groupby('Group'):
        group_cis[group] = []
        for pressure, pressure_group in group_df.groupby('Pressure'):
            bootstrap_samples = [
                pressure_group['Corrected Velocity'].sample(n=len(pressure_group), replace=True).median()
                for _ in range(n_iterations)
            ]
            ci_lower = np.percentile(bootstrap_samples, (100 - ci_percentile) / 2)
            ci_upper = np.percentile(bootstrap_samples, ci_percentile + (100 - ci_percentile) / 2)
            group_cis[group].append((pressure, ci_lower, ci_upper))

    # Plotting setup
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'≤50': 'tab:green', '>50': 'tab:orange', '<120': 'tab:blue', '≥120': 'tab:red', 'M': 'tab:purple', 'F': 'tab:pink'}

    # Plot each group
    for group, cis in group_cis.items():
        pressures = [x[0] for x in cis]
        medians = [np.median([x[1], x[2]]) for x in cis]
        errors = [(m-c[1], c[2]-m) for m, c in zip(medians, cis)]
        ax.errorbar(pressures, medians, yerr=np.transpose(errors), fmt='-o', label=f'Group {group}', color=colors[group])
    
    # Calculate overlaps and store/print them
    overlaps = {}
    groups = list(group_cis.keys())
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            overlaps[(groups[i], groups[j])] = [
                calculate_overlap(group_cis[groups[i]][k][1:], group_cis[groups[j]][k][1:])
                for k in range(len(group_cis[groups[i]]))
            ]

    ax.set_xlabel('Pressure')
    ax.set_ylabel('Velocity (um/s)')
    ax.set_title(f'Median Corrected Velocity vs. Pressure with {ci_percentile}% Confidence Interval')
    ax.legend()

    plt.show()

    # Optionally print or return the overlaps
    print(overlaps)
    return overlaps


# Function to calculate median and bootstrap 95% CI
def calculate_median_ci(group, n_iterations=1000, ci_percentile=95, dimensionless = False):
    medians_dimless = []
    medians = []
    if 'Dimensionless Velocity' in group.columns:
        for _ in range(n_iterations):
            sample = resample(group['Dimensionless Velocity'])
            medians_dimless.append(np.median(sample))
        lower = np.percentile(medians_dimless, (100 - ci_percentile) / 2)
        upper = np.percentile(medians_dimless, 100 - (100 - ci_percentile) / 2)
        median_dimless = np.median(group['Dimensionless Velocity'])
        
    else:
        for _ in range(n_iterations):
            sample = resample(group['Corrected Velocity'])
            medians.append(np.median(sample))
        lower = np.percentile(medians, (100 - ci_percentile) / 2)
        upper = np.percentile(medians, 100 - (100 - ci_percentile) / 2)
        median = np.median(group['Corrected Velocity'])
    if dimensionless:
        return pd.Series({'Median Dimensionless Velocity': median_dimless, 'CI Lower Bound': lower, 'CI Upper Bound': upper})
    else:
        return pd.Series({'Median Velocity': median, 'CI Lower Bound': lower, 'CI Upper Bound': upper})


def calculate_metrics(velocities):
    # Remove NaN values from the velocities
    velocities = velocities.dropna()
    metrics = {}
    metrics['std_dev'] = np.std(velocities)
    metrics['skewness'] = skew(velocities)
    metrics['kurtosis'] = kurtosis(velocities, fisher=False)  # Fisher=False for Pearson's definition of kurtosis
    metrics['peakiness'] = max(velocities) / metrics['std_dev']  # Example definition for peakiness
    metrics['coeff_variation'] = metrics['std_dev'] / np.mean(velocities)
    return metrics

# Function to create a color map for a given column
def create_color_map(df, column, cmap='viridis'):
    unique_values = sorted(df[column].unique())
    colors = sns.color_palette(cmap, len(unique_values))  
    return dict(zip(unique_values, colors))

def empirical_cdf_fn(data):
    """Create a function to calculate the CDF for any value in the dataset."""
    # Sort the data and calculate the CDF values
    sorted_data = np.sort(data)
    cdf_values = np.arange(1, len(data) + 1) / len(data)
    
    # Return a function that calculates the CDF for a given x
    def cdf_function(x):
        # If x is a single value, np.searchsorted will return a scalar; if x is an array, it will return an array
        return np.interp(x, sorted_data, cdf_values, left=0, right=1)
    
    return cdf_function

def plot_box_whisker_pressure(df, variable='Age', log_scale=False):
    """
    Plot box and whisker plots for the variable of interest grouped by Age or SYS_BP at each pressure level.

    Args:
        df (DataFrame): the DataFrame to be plotted
        variable (str): the variable of interest to group by, 'Age' or 'SYS_BP'
    
    Returns:
        0 if successful
    """
    # Set up the grouping based on the variable of interest
    if variable == 'Age':
        df['Group'] = np.where(df['Age'] <= 50, '≤50', '>50')
        hue = 'Group'
    else:
        df['Group'] = np.where(df['SYS_BP'] < 120, '<120', '≥120')
        hue = 'Group'

    # Plotting
    plt.figure(figsize=(10, 6))
    box_plot = sns.boxplot(x='Pressure', y='Corrected Velocity', hue=hue, data=df, showfliers=False)
    plt.title(f'Box and Whisker Plot of Velocity by Pressure and {variable} Group')
    plt.xlabel('Pressure')
    plt.ylabel('Velocity (um/s)')
    plt.legend(title=f'{variable} Group')

    # Set y-axis to logarithmic scale if specified
    if log_scale:
        box_plot.set_yscale('log')

    plt.show()
    return 0

def compare_participants(df1, df2):
    """ 
    Compare the participants in two DataFrames and return a list of participants that are in one DataFrame 
    but not the other, or have different counts in the two DataFrames.

    Args:
        df1 (DataFrame): the first DataFrame
        df2 (DataFrame): the second DataFrame
    
    Returns:
        
    """
    # Group by 'Participant' and 'Capillary' and count the number of rows for each group
    df1_grouped = df1.groupby(['Participant', 'Video', 'Capillary']).size()
    df2_grouped = df2.groupby(['Participant', 'Video', 'Capillary']).size()

    # Group by 'Participant' and count the total number of rows for each participant
    df1_participant_counts = df1['Participant'].value_counts()
    df2_participant_counts = df2['Participant'].value_counts()

    different_rows = []
    different_participants = []


    for row in df1_grouped.index:
        if row not in df2_grouped or df1_grouped[row] != df2_grouped[row]:
            different_rows.append(row)

    for row in df2_grouped.index:
        if row not in df1_grouped:
            different_rows.append(row)
    
    # Check for different total counts for participants
    for participant in df1_participant_counts.index:
        if participant not in df2_participant_counts or df1_participant_counts[participant] != df2_participant_counts[participant]:
            different_participants.append(participant)

    for participant in df2_participant_counts.index:
        if participant not in df1_participant_counts:
            different_participants.append(participant)

    return different_rows, different_participants

def plot_pca(df):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    features = ['Velocity', 'Pressure', 'SYS_BP']
    x = df.loc[:, features].values
    x = StandardScaler().fit_transform(x)

    # PCA
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)

    # Create a DataFrame with the PCA results
    principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
    principalDf['Age'] = df['Age']  # Add the age column

    # Plotting
    plt.figure(figsize=(10,8))
    scatter = sns.scatterplot(x='principal component 1', y='principal component 2', hue='Age', data=principalDf, palette='viridis')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('2 Component PCA Colored by Age')
    plt.colorbar(scatter,label='Age')
    plt.show()
    return 0
def plot_velocity_vs_pressure(df, hue = 'Age'):
    # plot velocity vs diameter scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Diameter', y='Corrected Velocity',hue = hue, data=df)
    plt.title('Diameter vs Velocity')
    plt.xlabel('Diameter')
    plt.ylabel('Velocity')
    plt.show()
    return 0
def plot_pressure_vs_diameter(df, hue = 'Age'):
    # Scatter plot for Age vs Velocity
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Pressure', y='Corrected Velocity', hue = hue, data=df)
    plt.title('Pressure vs Velocity')
    plt.xlabel('Pressure')
    plt.ylabel('Velocity')
    plt.show()
    return 0
def plot_velocity_vs_diameter(df, hue = 'Age'):
    # Create a 2x3 subplot figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Iterate over each pressure level
    for i, pressure in enumerate(df['Pressure'].unique()):
        # Filter the data for the current pressure level
        df_pressure = df[df['Pressure'] == pressure]
        # make all velocities between 0 and 1 equal to 0
        df_pressure['Corrected Velocity'] = df_pressure['Corrected Velocity'].apply(lambda x: 0 if x < 10 else x)
        
        # Select the subplot for the current pressure level
        ax = axes[i // 3, i % 3]
        
        # Scatter plot for diameter vs velocity
        sns.scatterplot(x='Diameter', y='Corrected Velocity', hue='Age', data=df_pressure, ax=ax)
        # set x axis to 0-100
        ax.set_xlim(10, 100)
        # set y axis to log scale
        ax.set_yscale('log')
        ax.set_xscale('log')

        
        # Set the title and labels for the subplot
        ax.set_title(f'Pressure: {pressure}')
        ax.set_xlabel('Diameter')
        ax.set_ylabel('Velocity')

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()

def plot_velocity_vs_diameter2(df, pressure=0.2, hue = 'Age'):
    # Create a 2x3 subplot figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Define the age groups
    age_groups = {
        'Under 24': (0, 23),
        '24-26': (24, 26),
        '27-32': (27, 32),
        '50-60': (50, 60),
        '61-65': (61, 65),
        'Over 66': (66, np.inf)
    }

    # Iterate over each age group
    for i in range(6):
        # Select the subplot for the current age group
        ax = axes[i // 3, i % 3]
        # Filter the data for the current age group
        min_age, max_age = list(age_groups.values())[i]
        df_age_group = df[(df['Age'] >= min_age) & (df['Age'] <= max_age) & (df['Pressure'] == pressure)]
        # make all velocities between 0 and 1 equal to 0
        df_age_group['Corrected Velocity'] = df_age_group['Corrected Velocity'].apply(lambda x: 10 if x < 10 else x)
        
        # Scatter plot for diameter vs velocity
        sns.scatterplot(x='Diameter', y='Corrected Velocity', hue='Age', data=df_age_group, ax=ax)
        # set x axis to 0-100
        ax.set_xlim(10, 100)
        # set y axis to log scale
        ax.set_yscale('log')
        ax.set_xscale('log')

        # Set the title and labels for the subplot
        ax.set_title(f'Age: {min_age}-{max_age}')
        ax.set_xlabel('Diameter')
        ax.set_ylabel('Velocity')

    
    # Adjust the spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()

def plot_average_viscocities(summary_df_dimless):
    viscocities = pd.DataFrame(columns=['Participant', 'Viscocity'])
    for participant in summary_df_dimless['Participant'].unique():
        participant_df = summary_df_dimless[summary_df_dimless['Participant'] == participant]
        viscocity = participant_df['Viscocity'].mean()
        viscocities = pd.concat([viscocities, pd.DataFrame({'Participant': [participant], 'Viscocity': [viscocity]})])
    # plot
    plt.figure(figsize=(10, 6))
    plt.bar(viscocities['Participant'], viscocities['Viscocity'])
    plt.xlabel('Participant')
    plt.ylabel('Viscocity')
    plt.title('Average Viscocity for Each Participant')
    plt.show()

""" Working on velocity normalization here """
def calc_nu_star(diameter):
    """ Calculates in-vivo viscocity coefficient
    Args:
        diameter (float): diameter of the capillary in micrometers

    Returns:
        nu_star (float): in-vivo viscocity coefficient
    """
    nu_star = (6*np.exp(-0.0855*diameter))+3.2-(2.44*np.exp(-0.06*(diameter**0.645)))
    return nu_star

def calc_constant(diameter):
    """ Calculates constant for the equation
    Args:
        diameter (float): diameter of the capillary in micrometers

    Returns:
        constant (float): constant value for the equation
    """
    weird_term = 1/(1+(10**(-11)*diameter**12))
    constant = (0.8+np.exp(-0.075*diameter))*(-1+weird_term)+ weird_term
    return constant

def calc_viscocity(diameter, nu_star, hemocrit = 0.45, constant = 0.1):
    """ Calculates in-vivo velocity based on Resistance to blood flow in microvessels in vivo. by Pries et al.
    
    Args:
        diameter (float): diameter of the capillary in micrometers
        nu_star (float): in-vivo viscocity coefficient
        hemocrit (float): hemocrit level of the blood
        constant (float): constant value for the equation

    Returns:
        viscocity (float): in-vivo viscocity
    """
    second_term = (diameter/(diameter-1.1))**2
    first_term = 1 + (nu_star-1)*((1-hemocrit)**constant -1)* second_term /((1-0.45)**constant -1) 
    viscocity = first_term*second_term  
    return viscocity

def calc_no_dim_v_divisor(velocity, diameter, viscocity):  
    """ Calculates the non-dimensionalized velocity divisor (characteristic velocity)
    Args:
        velocity (float): velocity of the blood
        diameter (float): diameter of the capillary in micrometers
        viscocity (float): in-vivo viscocity

    Returns:
        no_dim_v (float): non-dimensionalized velocity
    """
    no_dim_v_divisor = np.sqrt(np.abs(velocity*viscocity/diameter)) # maybe need to include log in here? or elsewhere?
    return no_dim_v_divisor

def add_dimensionless_velocity(summary_df):
    # Calculate dimensionless velocity for each row in the summary_df
    velocities = summary_df['Corrected Velocity']
    diameters = summary_df['Diameter']
    # print participant, video, capillary of nan diameter
    print(summary_df[summary_df['Diameter'].isna()][['Participant', 'Video', 'Capillary']])
    # drop row if diameter= nan
    summary_df = summary_df[~summary_df['Diameter'].isna()]
    # print velocities for all rows with zero diameter
    #TODO: make velocities with zero diameter into zero. 
    constants = calc_constant(diameters)
    viscocities = calc_viscocity(diameters, calc_nu_star(diameters), constant = constants)
    summary_df['Dimensionless Velocity Divisor'] = calc_no_dim_v_divisor(velocities, diameters, viscocities)
    summary_df['Viscocity'] = viscocities
    # print number of nan viscocities, corrected velocities, and diameters
    print(f'Number of nan velocities: {summary_df["Corrected Velocity"].isna().sum()}')
    print(f'Number of nan diameters: {summary_df["Diameter"].isna().sum()}')
    print(f'Number of nan viscocities: {summary_df["Viscocity"].isna().sum()}')
    print(f'Number of nan dimensionless velocity divisors: {summary_df["Dimensionless Velocity Divisor"].isna().sum()}')
    # print number of zero velocities, visc etc
    print(f'Number of zero velocities: {len(summary_df[summary_df["Corrected Velocity"] == 0])}')
    print(f'Number of zero diameters: {len(summary_df[summary_df["Diameter"] == 0])}')
    print(f'Number of zero viscocities: {len(summary_df[summary_df["Viscocity"] == 0])}')
    print(f'Number of zero dimensionless velocity divisors: {len(summary_df[summary_df["Dimensionless Velocity Divisor"] == 0])}')

    summary_df['Dimensionless Velocity'] =  summary_df['Corrected Velocity'] /summary_df['Dimensionless Velocity Divisor']
    # make all rows with zero diameter have zero velocity
    summary_df['Dimensionless Velocity'] = summary_df.apply(lambda x: 0 if x['Diameter'] == 0 else x['Dimensionless Velocity'], axis = 1)
    # make all rows with zero 'Corrected Velocity' have zero 'Dimensionless Velocity'
    summary_df['Dimensionless Velocity'] = summary_df.apply(lambda x: 0 if x['Corrected Velocity'] == 0 else x['Dimensionless Velocity'], axis = 1)
    
    #TODO go here
     # print number of nan viscocities, corrected velocities, and diameters
    print(f'Number of nan velocities: {summary_df["Corrected Velocity"].isna().sum()}')
    print(f'Number of nan diameters: {summary_df["Diameter"].isna().sum()}')
    print(f'Number of nan viscocities: {summary_df["Viscocity"].isna().sum()}')
    print(f'Number of nan dimensionless velocity divisors: {summary_df["Dimensionless Velocity Divisor"].isna().sum()}')
    # print number of zero velocities, visc etc
    print(f'Number of zero velocities: {len(summary_df[summary_df["Corrected Velocity"] == 0])}')
    print(f'Number of zero diameters: {len(summary_df[summary_df["Diameter"] == 0])}')
    print(f'Number of zero viscocities: {len(summary_df[summary_df["Viscocity"] == 0])}')
    print(f'Number of zero dimensionless velocity divisors: {len(summary_df[summary_df["Dimensionless Velocity Divisor"] == 0])}')

    # Calculate median dimensionless velocity for each video within each participant:
    summary_df['Median Dimensionless Velocity'] = summary_df.groupby(['Participant', 'Video'])['Dimensionless Velocity'].transform('median')
    return summary_df

""" End work on velocity normalization """

def plot_loc_histograms(df, variable, metrics = False):
    if variable == 'Age':
        point_variable = 'SYS_BP'
    else:
        point_variable = 'Age'

    # Create color map for 'Age' and 'SYS_BP'
    variable_color_map = create_color_map(df, variable)
    point_color_map = create_color_map(df, point_variable)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6), constrained_layout=True)
    num_bins = 5  # Specify the number of bins for the histograms
    
    # Create a unique identifier for each participant-location combination
    df['Participant_Location'] = df['Participant'] + "-" + df['Location']
    unique_ids = df['Participant_Location'].unique()
    
    # Compute median values for the histogram variable for each participant-location
    median_values_hist = df.groupby('Participant_Location')[variable].median()
    median_values_point = df.groupby('Participant_Location')[point_variable].median()

    # Create a mapping for x-axis positions
    x_positions = {id: index for index, id in enumerate(unique_ids)}

    for id in unique_ids:
        participant_data = df[df['Participant_Location'] == id]
        x_position = x_positions[id]

        # Calculate bins and frequencies
        velocities = participant_data['Corrected Velocity']
        bins = np.linspace(velocities.min(), velocities.max(), num_bins + 1)
        bin_indices = np.digitize(velocities, bins) - 1  # Bin index for each velocity

        # Normalize the bar heights
        total_measurements = len(velocities)
        bin_heights = np.array([np.sum(bin_indices == bin_index) for bin_index in range(num_bins)]) / total_measurements

        # Get the median value for the histogram variable
        hist_attribute_median = median_values_hist[id]

        # Plot bars for each bin
        for bin_index, bar_height in enumerate(bin_heights):
            if bar_height == 0:
                continue
            color = variable_color_map[hist_attribute_median]
            ax.bar(x_position + (bin_index - num_bins / 2) * 0.1, bar_height, color=color, width=0.1, align='center')

    # Customize the plot
    ax.set_xlabel('Participant-Location')
    ax.set_ylabel(f'Frequency of velocity')
    ax.set_title(f'Histogram of Velocities by Participant and Location\nColored by {variable}')
    
    # Secondary y-axis for the points
    ax2 = ax.twinx()
    for id in unique_ids:
        x_position = x_positions[id]
        point_attribute_median = median_values_point[id]
        point_color = point_color_map[point_attribute_median]
        ax2.plot(x_position, point_attribute_median, 'X', color='red', markersize=10)

    ax2.set_ylabel(f'{point_variable} Value')

    # Set x-ticks to be the participant-location names
    ax.set_xticks(list(x_positions.values()))
    ax.set_xticklabels(list(x_positions.keys()), rotation=45)

    # Legends
    hist_legend_elements = [Patch(facecolor=color, edgecolor='gray', label=label) for label, color in variable_color_map.items()]
    ax.legend(handles=hist_legend_elements, title=variable, bbox_to_anchor=(1.05, 1), loc='upper left')

    point_legend_elements = [Patch(facecolor='red', edgecolor='red', label=point_variable)]
    ax2.legend(handles=point_legend_elements, title=point_variable, bbox_to_anchor=(1.15, 0.9), loc='upper left')

    plt.show()
    return 0
def plot_histograms(df, variable = 'Age', diam_slice = None, normalize_bins = 'Total', gradient = True):
    """
    Plot histograms of the velocities for each participant, colored by the specified variable.

    Args:
        df (DataFrame): the DataFrame to be plotted
        variable (str): the variable to color the histograms by
        diam_slice (str): the slice of the DataFrame to be plotted. Default is None. Options are 'smaller' and 'larger'
        normalize_bins (str): the method for normalizing the bin heights. Options are 'Total' and 'Participant'
    Returns:
        0 if successful
    """
    
    #------------------------Histograms------------------------
    if variable == 'Age':
        point_variable = 'SYS_BP'
    else:
        point_variable = 'Age'

    if gradient:
        # Determine the range of the variable for the gradient
        variable_range = (df[variable].min(), df[variable].max())
        norm = mcolors.Normalize(vmin=variable_range[0], vmax=variable_range[1])
        scalar_map = ScalarMappable(norm=norm, cmap='viridis')  
    else:
        # Create color map for 'Age' and 'SYS_BP'
        variable_color_map = create_color_map(df, variable)
        point_color_map = create_color_map(df, point_variable)

    # Calculate the median 'variable' for each participant and sort them
    median_variable_per_participant = df.groupby('Participant')[variable].median().sort_values()
    participant_order = {participant: index for index, participant in enumerate(median_variable_per_participant.index)}

    fig, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
    num_bins = 5  # Specify the number of bins for the histograms
    
    # Compute median values for the histogram variable for each participant
    median_values_hist = df.groupby('Participant')[variable].median()
    median_values_point = df.groupby('Participant')[point_variable].median()

    for participant in participant_order:
        participant_data = df[df['Participant'] == participant]
        participant_index = participant_order[participant]
        participant_velocities = participant_data['Corrected Velocity']
        max_velocity = participant_velocities.max()
        if max_velocity > 2000:
            bins = [-0.0001, 5, calc_vel_bins(15), calc_vel_bins(30), calc_vel_bins(45), calc_vel_bins(60), calc_vel_bins(75), 2000, max_velocity+1]
        else:
            bins = [-0.0001, 5, calc_vel_bins(15), calc_vel_bins(30), calc_vel_bins(45), calc_vel_bins(60), calc_vel_bins(75), 2000]
        
        # bins = np.linspace(velocities.min(), velocities.max(), num_bins + 1)
        # bins = [0, 5, 55, 161, df['Corrected Velocity'].max()]
        num_bins = len(bins) - 1
        bin_indices = np.digitize(participant_velocities, bins) - 1  # Bin index for each velocity

        # Normalize the bar heights to the total number of measurements for the participant
        total_measurements = len(participant_velocities)
        bin_heights = np.array([np.sum(bin_indices == bin_index) for bin_index in range(num_bins)])#/ total_measurements

        # Get the median value for the histogram variable for the participant
        hist_attribute_median = median_values_hist[participant]

        # Plot bars for each bin
        for bin_index, bar_height in enumerate(bin_heights):
            if bar_height == 0:
                continue
            if gradient:
                color = scalar_map.to_rgba(hist_attribute_median)
            else:
                color = variable_color_map[hist_attribute_median]
            ax.bar(participant_index + (bin_index - num_bins / 2) * 0.1, bar_height,
                   color=color, width=0.1, align='center')

    # Customize the plot
    ax.set_xlabel('Participant')
    ax.set_ylabel(f'Frequency of velocity')
    if diam_slice == 'smaller':
        ax.set_title(f'Histogram of Velocities by Participant\nColored by {variable} \n(Smaller Diameters)')
    elif diam_slice == 'larger':
        ax.set_title(f'Histogram of Velocities by Participant\nColored by {variable} \n(Larger Diameters)')
    else:
        ax.set_title(f'Histogram of Velocities by Participant\nColored by {variable}')
    
    # Create secondary y-axis for the points
    ax2 = ax.twinx()
    for participant in participant_order:
        participant_data = df[df['Participant'] == participant]
        participant_index = participant_order[participant]
        
        # Get the attribute value for the points
        point_attribute_median = median_values_point[participant]

        # if point_attribute_median has a decimal, round to the nearest whole number
        if point_attribute_median % 1 >= 0.5:
            point_attribute_median = np.ceil(point_attribute_median)
        else:
            point_attribute_median = np.floor(point_attribute_median)
        
        # Plot the point
        ax2.plot(participant_index, point_attribute_median, 'o', color='red', markersize=5)         # could make this point color
    
    ax2.set_ylabel(f'{point_variable} Value')

    # Set x-ticks to be the participant names
    ax.set_xticks(list(participant_order.values()))
    ax.set_xticklabels(list(participant_order.keys()))

    if gradient:
        hist_legend_elements = [Patch(facecolor=scalar_map.to_rgba(value), edgecolor='gray', label=value)
                                 for value in median_variable_per_participant.values]
    else:
        # Create a legend for the attribute
        hist_legend_elements = [Patch(facecolor=color, edgecolor='gray', label=label)
                       for label, color in variable_color_map.items()]
    ax.legend(handles=hist_legend_elements, title=variable, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Create a legend for the points
    point_legend_elements = [Patch(facecolor='red', edgecolor='red', label=point_variable)]
    ax2.legend(handles=point_legend_elements, title=point_variable, bbox_to_anchor=(1.15, 0.9), loc='upper left')
    plt.show()
    return 0
def plot_velocities(participant_df, write=False):
    participant_df = participant_df.copy()
    # Assuming there's a 'Location' column or similar to distinguish the same capillary used in different contexts
    # If not, you might need to create a composite key or a unique identifier based on your specific needs
    participant_df['Capillary_Location'] = participant_df['Capillary'].astype(str) + '_' + participant_df['Location'].astype(str)
    
    # Group the data by the new 'Capillary_Location'
    grouped_df = participant_df.groupby('Capillary_Location')
    # Get the unique capillary-location identifiers
    capillary_locations = participant_df['Capillary_Location'].unique()
    participant = participant_df['Participant'].unique()[0]

    # Calculate the number of plots and rows needed
    num_plots = len(capillary_locations)
    num_rows = (num_plots + 3) // 4

    # Adjust the height here; you can increase the multiplier as needed
    # For example, use 3 or 4 instead of 2 to make each plot taller
    fig_height = 3 * num_rows  # Increase the multiplier to give more space

    # Create subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=4, figsize=(20, fig_height), sharey=True, sharex=True)
    axes = axes.flatten()

    for i, capillary_location in enumerate(capillary_locations):
        capillary_data = grouped_df.get_group(capillary_location).copy()
        ax = axes[i]

        # Create 'Up/Down' column based on pressure changes
        capillary_data.loc[:, 'Up/Down'] = 'Up'
        max_pressure = capillary_data['Pressure'].max()
        max_index = capillary_data['Pressure'].idxmax()
        capillary_data.loc[max_index:, 'Up/Down'] = 'Down'

        # Separate data for plotting
        data_up = capillary_data[capillary_data['Up/Down'] == 'Up']
        data_down = capillary_data[capillary_data['Up/Down'] == 'Down']

        # Plot data
        ax.plot(data_up['Pressure'], data_up['Corrected Velocity'], marker='o', linestyle='-', label='Increase in Pressure')
        ax.plot(data_down['Pressure'], data_down['Corrected Velocity'], color='purple', marker='o', linestyle='-', label='Decrease in Pressure')
        
        ax.set_xlabel('Pressure (psi)')
        ax.set_ylabel('Velocity (um/s)')
        ax.set_title(f'{participant} {capillary_location}')
        ax.grid(True)
        ax.legend()

    # Remove unused subplots
    for i in range(num_plots, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    if write:
        plt.savefig(f'C:\\Users\\gt8ma\\capillary-flow\\results\\{participant}_fav_cap_v.png', dpi = 600)
        plt.close()
    else:
        plt.show()

    return 0
def plot_caps_by_size(summary_df):
    """
    Plot the histograms of velocities for the 9 smallest diameter participants and the 9 largest diameter participants.

    Args:
        summary_df (DataFrame): the DataFrame to be plotted
    Returns:
        0 if successful
    """
    # find the median diameter for each participant
    median_diameter_per_participant = summary_df.groupby('Participant')['Diameter'].median().sort_values()
    sorted_participant_indices = {participant: index for index, participant in enumerate(median_diameter_per_participant.index)}

    # choose the 9 smallest diameter participants: (0-8)
    smaller_diameters = list(sorted_participant_indices.keys())[0:9]
    larger_diameters = list(sorted_participant_indices.keys())[9:19]

    # slice summary_df into smaller and larger diameter dfs
    smaller_diameter_df = summary_df[summary_df['Participant'].isin(smaller_diameters)]
    larger_diameter_df = summary_df[summary_df['Participant'].isin(larger_diameters)]

    # plot the histogram of velocities for each participant in sliced dfs
    main(smaller_diameter_df, 'Age', diam_slice = 'smaller')
    main(larger_diameter_df, 'Age', diam_slice = 'larger')
    main(smaller_diameter_df, 'SYS_BP', diam_slice = 'smaller')
    main(larger_diameter_df, 'SYS_BP', diam_slice = 'larger')
    return 0
def plot_median_diameter(summary_df):
    """
    Plot the median diameter for each participant.

    Args:
        summary_df (DataFrame): the DataFrame to be plotted, containing 'Participant', 'Diameter', and 'Age' columns
    Returns:
        0 if successful
    """
    # find the median diameter for each participant
    median_diameter_per_participant = summary_df.groupby('Participant')['Diameter'].median().sort_values()
    sorted_participant_indices = {participant: index for index, participant in enumerate(median_diameter_per_participant.index)}

    # plot the median diameter for each participant
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
    # make blue bars hollow and the edges thick
    ax2 = ax.twinx()
    ax.bar(sorted_participant_indices.values(), median_diameter_per_participant.values, width=0.5)
    # add a red dot for the age of each participant on second y-axis
    ax2.plot(sorted_participant_indices.values(), summary_df.groupby('Participant')['Age'], '.', color='black', markersize=10)
    ax.set_xlabel('Participant')
    ax.set_ylabel('Median Diameter')
    ax2.set_ylabel('Age')
    ax.set_title('Median Diameter for Each Participant')
    ax.set_xticks(list(sorted_participant_indices.values()), list(sorted_participant_indices.keys()))
    plt.show()
    return 0
def compile_metadata():
    metadata_folder = 'C:\\Users\\gt8mar\\capillary-flow\\metadata'
    # Read the metadata files if they are csvs
    metadata_files = [f for f in os.listdir(metadata_folder) if f.endswith('.xlsx')]
    metadata_dfs = [pd.read_excel(os.path.join(metadata_folder, f)) for f in metadata_files]
    metadata_df = pd.concat(metadata_dfs)

    # make slice of metadata_df with only bp measurements
    non_bp_metadata = metadata_df[~metadata_df['Video'].str.contains('bp')]
   
    # add 'loc' and a leading zero to the location column
    non_bp_metadata['Location'] = 'loc' + non_bp_metadata['Location'].astype(str).str.zfill(2)

    # Convert 'Video' identifiers to integers for comparison
    non_bp_metadata['VideoID'] = non_bp_metadata['Video'].str.extract('(\d+)').astype(int)

    # remove all part09 videos greater than vid59:
    non_bp_metadata = non_bp_metadata[~((non_bp_metadata['Participant'] == 'part09') & (non_bp_metadata['VideoID'] > 59))]

    # keep only participant, date, location, and video columns
    non_bp_metadata = non_bp_metadata[['Participant', 'Date', 'Location', 'Video']]
    return non_bp_metadata

def check_inserted_rows(summary_df):
     # if the row has 'inserted' in the notes column, that means that the area is the same as the original area from the 'Capillary' column
    condition_inserted = summary_df['Notes'].str.contains('inserted')
    # print the rows that have 'inserted' in the notes column
    print(summary_df[condition_inserted][['Participant', 'Date', 'Location', 'Video', 'Capillary', 'Capillary_new', 'Area']])

    # print all part26 loc01 rows
    print(summary_df[(summary_df['Participant'] == 'part26') & (summary_df['Location'] == 'loc01') & (summary_df['Video'] == 'vid05')][['Participant', 'Date', 'Location', 'Video', 'Capillary', 'Capillary_new', 'Area', 'Corrected Velocity', 'Diameter']])
    return 0
def handle_dotted_evac(summary_df):
    # Fill NaN values in the 'Notes' column with an empty string
    summary_df['Notes'] = summary_df['Notes'].fillna('')
    # Now it is time to handle the added rows in velocity_df

    # if the row has 'dotted' in the notes column, that means that the velocity and area are zero.
    # if the row has 'evac' in the notes column, that means that the velocity is zero and the area is zero.

    # Condition to find rows with 'NaN' in 'Area' and 'dotted' in 'Notes'
    condition = summary_df['Area'].isna() & summary_df['Notes'].str.contains('dotted')
    condition_evac = summary_df['Area'].isna() & summary_df['Notes'].str.contains('evac')
    summary_df.loc[condition, 'Area'] = 0
    summary_df.loc[condition, 'Diameter'] = 0
    summary_df.loc[condition, 'Corrected Velocity'] = 0
    summary_df.loc[condition, 'Centerline'] = 0
    summary_df.loc[condition_evac, 'Area'] = 0
    summary_df.loc[condition_evac, 'Diameter'] = 0
    summary_df.loc[condition_evac, 'Corrected Velocity'] = 0
    summary_df.loc[condition_evac, 'Centerline'] = 0
    return summary_df
def merge_vel_size(verbose=False):
    size_df = pd.read_csv('C:\\Users\\gt8mar\\capillary-flow\\results\\cap_diameters.csv')
    # velocity_df = pd.read_csv('C:\\Users\\gt8mar\\capillary-flow\\results\\velocities\\big_df - Copy.csv')
    velocity_df = pd.read_excel('C:\\Users\\gt8mar\\capillary-flow\\results\\big_df.xlsx')
    # velocity_df_old = pd.read_csv('C:\\Users\\gt8mar\\capillary-flow\\results\\velocities\\big_df.csv')
    metadata_df = compile_metadata()
    print(metadata_df.head)

    # modify size_df to remove all bp measurements
    print(f'size_df shape: {size_df.shape}')
    
    # remove 'bp' from the video column
    size_df['Video'] = size_df['Video'].str.replace('bp', '')

    # use outer merge to find the rows in size_df that are not bp measurements
    size_df = size_df.merge(metadata_df, on=['Participant', 'Date', 'Location', 'Video'], how='inner', indicator=False)
    print(f'new size_df shape: {size_df.shape}')
    print(size_df.head())

    different_rows, different_participants = compare_participants(size_df, velocity_df)

    # save for testing
    # size_df.to_csv('C:\\Users\\gt8mar\\capillary-flow\\size_test.csv', index=False)
    # velocity_df.to_csv('C:\\Users\\gt8mar\\capillary-flow\\velocity_test.csv', index=False)

    # # remove part22 and part23 from different rows
    # different_rows = [row for row in different_rows if row[0] != 'part22' and row[0] != 'part23']
    print(different_rows)
    print(different_participants)

    # pd.set_option('display.max_rows', None)

    # print(velocity_df[velocity_df['Participant'] == 'part15'][['Participant', 'Date', 'Location', 'Video', 'Capillary', 'Corrected Velocity']])
    # velocity_part15_shape = velocity_df[velocity_df['Participant'] == 'part15'].shape
    # print(f'Velocity df shape: {velocity_part15_shape}')
    # print(size_df[size_df['Participant'] == 'part15'][['Participant', 'Date', 'Location', 'Video', 'Capillary', 'Diameter']])
    # size_part15_shape = size_df[size_df['Participant'] == 'part15'].shape
    # print(f'Size df shape: {size_part15_shape}')

    # remove SYS_BP column from size_df
    size_df = size_df.drop(columns=['SYS_BP'])

    # Merge the DataFrames
    summary_df = pd.merge(size_df, velocity_df, how='outer',on=['Participant', 'Date', 'Location', 'Video', 'Capillary', 'Age'], indicator=True)
    
    summary_df = handle_dotted_evac(summary_df)

    if verbose: 
        check_inserted_rows(summary_df)
    
    # print any rows where area is NaN
    print("the following rows have NaN in the 'Area' column: ")
    print(summary_df[summary_df['Area'].isna()][['Participant', 'Date', 'Location', 'Video', 'Capillary', 'Area', 'Corrected Velocity', 'Diameter']])
    
    # make the 'Drop' column strings
    summary_df['Drop'] = summary_df['Drop'].astype(str)

    # remove row if 'drop' is in the Drop column
    summary_df = summary_df[~summary_df['Drop'].str.contains('drop')]

    # if there is a number in "Manual" column, replace "Corrected Velocity" with "Manual"
    summary_df['Corrected Velocity'] = summary_df['Manual'].fillna(summary_df['Corrected Velocity'])
    # if there is a number in "Manual Velocity" column, replace "Corrected Velocity" with "Manual Velocity"
    summary_df['Corrected Velocity'] = summary_df['Manual Velocity'].fillna(summary_df['Corrected Velocity'])
   
    # save summary_df to csv
    summary_df.to_csv('C:\\Users\\gt8mar\\capillary-flow\\summary_df_test.csv', index=False)
    return summary_df
def calc_vel_bins(theta):
    v_tan = np.tan(np.radians(theta))*2.44*(227.8/2)
    return v_tan

def plot_and_calculate_area(df, method='trapezoidal', plot = False, normalize = False, verbose=False):
    """
    Plot 'Pressure' vs. 'Corrected Velocity' from a DataFrame and calculate the area under the curve.
    
    Args:
        df (DataFrame): pressure and velocity data from a specific capillary location and up or down run. 
        method (str): The method to be used for the area calculation. Options are 'trapezoidal' and 'simpson'.
    
    Returns:
        area (float): The area under the curve calculated using the specified method.
    """
    df = df.copy()
    # Check if required columns are in the DataFrame
    if 'Pressure' not in df.columns or 'Corrected Velocity' not in df.columns:
        raise ValueError("DataFrame must contain 'Pressure' and 'Corrected Velocity' columns")
    
    # Normalization and hysteresis plotting
    if normalize:
        max_velocity = df['Corrected Velocity'].max()
        if max_velocity > 2000:
            bins = [-0.0001, 5, calc_vel_bins(15), calc_vel_bins(30), calc_vel_bins(45), calc_vel_bins(60), calc_vel_bins(75), 2000, max_velocity+1]
        else:
            bins = [-0.0001, 5, calc_vel_bins(15), calc_vel_bins(30), calc_vel_bins(45), calc_vel_bins(60), calc_vel_bins(75), 2000]
        bin_labels = range(len(bins)-1)  # Assigning a specific value to each bin
        df.loc[:, 'Velocity Binned'] = pd.cut(df['Corrected Velocity'].copy(), bins=bins, labels=bin_labels)
        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(df['Pressure'], df['Velocity Binned'])
            plt.title('Hysteresis Plot')
            plt.xlabel('Pressure')
            plt.ylabel('Binned Corrected Velocity')
            plt.legend()
            plt.grid(True)
            plt.show()
        if verbose:
            print(f'the binned velocities are: {df["Corrected Velocity"]} to {df["Velocity Binned"]}')
        # Area calculation    
        if method == 'trapezoidal':
            area = trapezoid(df['Velocity Binned'], df['Pressure'])
        elif method == 'simpson' and len(df) % 2 == 0:
            # Simpson's rule requires an even number of samples, adding a check
            print("Warning: Simpson's rule requires an even number of intervals. Adjusting by removing the last data point.")
            area = simps(df['Velocity Binned'][:-1], df['Pressure'][:-1])
        elif method == 'simpson':
            area = simps(df['Velocity Binned'], df['Pressure'])
        else:
            raise ValueError("Method must be either 'trapezoidal' or 'simpson'")
        if verbose:
            print(f"Calculated normalized area under the curve using {method} rule: {area}")
    else:
        if plot:
            # Plotting
            plt.figure(figsize=(10, 6))
            plt.plot(df['Pressure'], df['Corrected Velocity'], label='Corrected Velocity vs. Pressure', marker='o')
            plt.fill_between(df['Pressure'], df['Corrected Velocity'], alpha=0.2)
            plt.title('Corrected Velocity vs. Pressure')
            plt.xlabel('Pressure')
            plt.ylabel('Corrected Velocity')
            plt.legend()
            plt.grid(True)
            plt.show()
        
        # Area calculation
        if method == 'trapezoidal':
            area = trapezoid(df['Corrected Velocity'], df['Pressure'])
        elif method == 'simpson' and len(df) % 2 == 0:
            # Simpson's rule requires an even number of samples, adding a check
            print("Warning: Simpson's rule requires an even number of intervals. Adjusting by removing the last data point.")
            area = simps(df['Corrected Velocity'][:-1], df['Pressure'][:-1])
        elif method == 'simpson':
            area = simps(df['Corrected Velocity'], df['Pressure'])
        else:
            raise ValueError("Method must be either 'trapezoidal' or 'simpson'")
        if verbose:
            print(f"Calculated area under the curve using {method} rule: {area}")
    return area

def calc_average_pdf(df):
    pdfs = []
    for participant in df['Participant'].unique():
        # probability density function for each participant
        pdf_participant = df[df['Participant'] == participant]['Corrected Velocity'].value_counts(normalize=True).sort_index()
        pdfs.append(pdf_participant)
    # average pdf
    average_pdf = pd.concat(pdfs, axis=1).mean(axis=1)
    return average_pdf

def plot_densities(df, normalize = True):
    # Subset data into old vs young
    old_df = df[df['Age'] > 50]
    young_df = df[df['Age'] <= 50]

    if normalize:
        # insert function here like the one in plot_cdf
        average_pdf = calc_average_pdf(df)
        old_pdf = calc_average_pdf(old_df)
        young_pdf = calc_average_pdf(young_df)

        # Plot density
        sns.kdeplot(average_pdf, label='Entire Dataset', fill=True)
        sns.kdeplot(old_pdf, label='old', fill=True, alpha=0.5)
        sns.kdeplot(young_pdf, label='young', fill=True, alpha=0.5)
        plt.legend()
        plt.title('Normalized Density Plot of Entire Dataset vs. Subset')
        plt.show()
        return 0

    else: 

        # Plot density
        sns.kdeplot(df['Corrected Velocity'], label='Entire Dataset', fill=True)
        sns.kdeplot(old_df['Corrected Velocity'], label='old', fill=True, alpha=0.5)
        sns.kdeplot(young_df['Corrected Velocity'], label='young', fill=True, alpha=0.5)
        plt.legend()
        plt.title('Density Plot of Entire Dataset vs. Subset')
        plt.show()

        # Subset data into low BP vs high BP
        normBP_df = df[df['SYS_BP'] <= 120]
        highBP_df = df[df['SYS_BP'] > 120]
        print(f'the participants with high BP are: {highBP_df["Participant"].unique()}')

        # Plot density
        sns.kdeplot(df['Corrected Velocity'], label='Entire Dataset', fill=True)
        sns.kdeplot(highBP_df['Corrected Velocity'], label='high BP', fill=True, alpha=0.5)
        sns.kdeplot(normBP_df['Corrected Velocity'], label='normal', fill=True, alpha=0.5)
        plt.legend()
        plt.title('Density Plot of Entire Dataset vs. Subset')
        plt.show()

        # Plot density of old high BP vs young high BP vs old low BP vs young low BP
        old_highBP_df = old_df[old_df['SYS_BP'] > 120]
        young_highBP_df = young_df[young_df['SYS_BP'] > 120]
        old_normBP_df = old_df[old_df['SYS_BP'] <= 120]
        young_normBP_df = young_df[young_df['SYS_BP'] <= 120]

        # Plot density
        sns.kdeplot(df['Corrected Velocity'], label='Entire Dataset', fill=True)
        sns.kdeplot(old_highBP_df['Corrected Velocity'], label='old high BP', fill=True, alpha=0.5)
        sns.kdeplot(young_highBP_df['Corrected Velocity'], label='young high BP', fill=True, alpha=0.5)
        sns.kdeplot(old_normBP_df['Corrected Velocity'], label='old normal BP', fill=True, alpha=0.5)
        sns.kdeplot(young_normBP_df['Corrected Velocity'], label='young normal BP', fill=True, alpha=0.5)
        plt.legend()
        plt.title('Density Plot of Entire Dataset vs. Subset')
        plt.show()

        # compare high BP old vs young
        # Plot density
        sns.kdeplot(df['Corrected Velocity'], label='Entire Dataset', fill=True)
        sns.kdeplot(old_highBP_df['Corrected Velocity'], label='old high BP', fill=True, alpha=0.5)
        sns.kdeplot(young_highBP_df['Corrected Velocity'], label='young high BP', fill=True, alpha=0.5)
        plt.legend()
        plt.title('Density Plot of high BP participants')
        plt.show()

        # compare low BP old vs young
        # Plot density
        sns.kdeplot(df['Corrected Velocity'], label='Entire Dataset', fill=True)
        sns.kdeplot(old_normBP_df['Corrected Velocity'], label='old normal BP', fill=True, alpha=0.5)
        sns.kdeplot(young_normBP_df['Corrected Velocity'], label='young normal BP', fill=True, alpha=0.5)
        plt.legend()
        plt.title('Density Plot of normal BP participants')
        plt.show()
        return 0

def plot_densities_pressure(summary_df):
        # Subset data into old vs young
    old_df = summary_df[summary_df['Age'] > 50]
    young_df = summary_df[summary_df['Age'] <= 50]
    # Plot density
    sns.kdeplot(summary_df['Pressure'], label='Entire Dataset', fill=True)
    sns.kdeplot(old_df['Pressure'], label='old', fill=True, alpha=0.5)
    sns.kdeplot(young_df['Pressure'], label='young', fill=True, alpha=0.5)
    plt.legend()
    plt.title('Density Plot of Entire Dataset vs. Subset')
    plt.show()

    # Subset data into low BP vs high BP
    normBP_df = summary_df[summary_df['SYS_BP'] <= 120]
    highBP_df = summary_df[summary_df['SYS_BP'] > 120]
    print(f'the participants with high BP are: {highBP_df["Participant"].unique()}')

    # Plot density
    sns.kdeplot(summary_df['Pressure'], label='Entire Dataset', fill=True)
    sns.kdeplot(normBP_df['Pressure'], label='normal', fill=True, alpha=0.5)
    sns.kdeplot(highBP_df['Pressure'], label='high BP', fill=True, alpha=0.5)
    plt.legend()
    plt.title('Density Plot of Entire Dataset vs. Subset')
    plt.show()

    # Plot density of old high BP vs young high BP vs old low BP vs young low BP
    old_highBP_df = old_df[old_df['SYS_BP'] > 120]
    young_highBP_df = young_df[young_df['SYS_BP'] > 120]
    old_normBP_df = old_df[old_df['SYS_BP'] <= 120]
    young_normBP_df = young_df[young_df['SYS_BP'] <= 120]

    # Plot density
    sns.kdeplot(summary_df['Pressure'], label='Entire Dataset', fill=True)
    sns.kdeplot(old_highBP_df['Pressure'], label='old high BP', fill=True, alpha=0.5)
    sns.kdeplot(young_highBP_df['Pressure'], label='young high BP', fill=True, alpha=0.5)
    sns.kdeplot(old_normBP_df['Pressure'], label='old normal BP', fill=True, alpha=0.5)
    sns.kdeplot(young_normBP_df['Pressure'], label='young normal BP', fill=True, alpha=0.5)
    plt.legend()
    plt.title('Density Plot of Entire Dataset vs. Subset')
    plt.show()

    # compare high BP old vs young
    # Plot density
    sns.kdeplot(summary_df['Pressure'], label='Entire Dataset', fill=True)
    sns.kdeplot(old_highBP_df['Pressure'], label='old high BP', fill=True, alpha=0.5)
    sns.kdeplot(young_highBP_df['Pressure'], label='young high BP', fill=True, alpha=0.5)
    plt.legend()
    plt.title('Density Plot of high BP participants')
    plt.show()

    # compare low BP old vs young
    # Plot density
    sns.kdeplot(summary_df['Pressure'], label='Entire Dataset', fill=True)
    sns.kdeplot(old_normBP_df['Pressure'], label='old normal BP', fill=True, alpha=0.5)
    sns.kdeplot(young_normBP_df['Pressure'], label='young normal BP', fill=True, alpha=0.5)
    plt.legend()
    plt.title('Density Plot of normal BP participants')
    plt.show()
    return 0


def plot_densities_individual(summary_df, participant_df, participant):
    # Plot density
    sns.kdeplot(summary_df['Corrected Velocity'], label='Entire Dataset', fill=True)
    sns.kdeplot(participant_df['Corrected Velocity'], label=participant, fill=True, alpha=0.5)
    plt.legend()
    plt.title('Density Plot of Entire Dataset vs. Subset')
    plt.show()
    return 0

def plot_densities_pressure_individual(summary_df, participant_df, participant):
    # Plot density
    sns.kdeplot(summary_df['Pressure'], label='Entire Dataset', fill=True)
    sns.kdeplot(participant_df['Pressure'], label=participant, fill=True, alpha=0.5)
    plt.legend()
    plt.title('Density Plot of Entire Dataset vs. Subset')
    plt.show()
    return 0

def plot_hist_pressure(summary_df, normalize = False, density = False):
    # Subset data into old vs young
    old_df = summary_df[summary_df['Age'] > 50]
    young_df = summary_df[summary_df['Age'] <= 50]
    # normalize by the number of participants
    plt.hist(summary_df['Pressure'], bins=[0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,10], label='Entire Dataset', alpha = 0.5, density=density)
    plt.hist(old_df['Pressure'], bins=[0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,10], label='Old Participants', alpha = 0.5, density=density)
    plt.hist(young_df['Pressure'], bins=[0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,10], label='Young Participants', alpha = 0.5, density=density)
    plt.legend()
    plt.title('Pressure Histogram')
    plt.show()

    # Subset data into low BP vs high BP
    normBP_df = summary_df[summary_df['SYS_BP'] <= 120]
    highBP_df = summary_df[summary_df['SYS_BP'] > 120]
    plt.hist(summary_df['Pressure'], bins=[0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,10], label='Entire Dataset', alpha = 0.5, density=density)
    plt.hist(highBP_df['Pressure'], bins=[0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,10], label='High BP', alpha = 0.5, density=density)
    plt.hist(normBP_df['Pressure'], bins=[0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,10], label='Normal BP', alpha = 0.5, density=density)
    plt.legend()
    plt.title('Pressure Histogram')
    plt.show()
    return 0

def plot_hist_specific_pressure(df, pressure, density = False, hist = True):
    specific_velocities = df[df['Pressure'] == pressure]
    old_df = specific_velocities[specific_velocities['Age'] > 50]
    young_df = specific_velocities[specific_velocities['Age'] <= 50]
    if hist:
        plt.hist(specific_velocities['Corrected Velocity'], bins=20, label=f'Pressure: {pressure}', alpha = 0.5, density=density)
        plt.hist(old_df['Corrected Velocity'], bins=20, label=f'Old Participants', alpha = 0.5, density=density)
        plt.hist(young_df['Corrected Velocity'], bins=20, label=f'Young Participants', alpha = 0.5, density=density)
    else:
        sns.kdeplot(specific_velocities['Corrected Velocity'], label=f'Pressure: {pressure}', fill=True)
        sns.kdeplot(old_df['Corrected Velocity'], label=f'Old Participants', fill=True)
        sns.kdeplot(young_df['Corrected Velocity'], label=f'Young Participants', fill=True)
    plt.legend()
    plt.title(f'Velocity Histogram at Pressure: {pressure}')
    plt.show()

    # Subset data into low BP vs high BP
    normBP_df = specific_velocities[specific_velocities['SYS_BP'] <= 120]
    highBP_df = specific_velocities[specific_velocities['SYS_BP'] > 120]
    plt.hist(specific_velocities['Corrected Velocity'], bins=20, label=f'Pressure: {pressure}', alpha = 0.5, density=density)
    plt.hist(normBP_df['Corrected Velocity'], bins=20, label=f'Normal BP', alpha = 0.5, density=density)
    plt.hist(highBP_df['Corrected Velocity'], bins=20, label=f'High BP', alpha = 0.5, density=density)
    plt.legend()
    plt.title(f'Velocity Histogram at Pressure: {pressure}')
    plt.show()
    return 0

def plot_hist_comp_pressure(df, normalize = False, density = False, hist = True, fill = False):
    low_p = df[df['Pressure'] == 0.2]
    med_p = df[df['Pressure'] == 0.8]
    high_p = df[df['Pressure'] == 1.2]

    # Plot density
    if hist:
        plt.hist(low_p['Corrected Velocity'], bins=20, label='0.2 psi', alpha = 0.5, density=density)
        plt.hist(med_p['Corrected Velocity'], bins=20, label='0.8 psi', alpha = 0.5, density=density)
        plt.hist(high_p['Corrected Velocity'], bins=20, label='1.2 psi', alpha = 0.5, density=density)
    else:
        sns.kdeplot(low_p['Corrected Velocity'], label='0.2 psi', fill=True)
        sns.kdeplot(med_p['Corrected Velocity'], label='0.8 psi', fill=True)
        sns.kdeplot(high_p['Corrected Velocity'], label='1.2 psi', fill=True)
    plt.legend()
    plt.title('Density Plot of different pressures')
    plt.show()

    # make subset of old and young participants
    old_low_p = low_p[low_p['Age'] > 50]
    young_low_p = low_p[low_p['Age'] <= 50]
    old_med_p = med_p[med_p['Age'] > 50]
    young_med_p = med_p[med_p['Age'] <= 50]
    old_high_p = high_p[high_p['Age'] > 50]
    young_high_p = high_p[high_p['Age'] <= 50]

    # # Plot density old first
    # if hist:
    #     plt.hist(low_p['Corrected Velocity'], bins=20, label='0.2 psi', alpha = 0.5, density=density, color = 'C0', histtype='step')
    #     plt.hist(old_low_p['Corrected Velocity'], bins=20, label='old 0.2 psi', alpha = 0.5, density=density, color = 'C1', histtype='step')
    #     plt.hist(low_p['Corrected Velocity'], bins=20, label='0.8 psi', alpha = 0.5, density=density, color = 'C2')
    #     plt.hist(old_med_p['Corrected Velocity'], bins=20, label='old 0.8 psi', alpha = 0.5, density=density, color = 'C3')
    #     plt.hist(low_p['Corrected Velocity'], bins=20, label='1.2 psi', alpha = 0.5, density=density, color = 'C4', histtype='step')
    #     plt.hist(old_high_p['Corrected Velocity'], bins=20, label='old 1.2 psi', alpha = 0.5, density=density, color = 'C5', histtype='step')
    #     # plt.hist(young_low_p['Corrected Velocity'], bins=20, label='young 0.2 psi', alpha = 0.5, density=density)
    # else:
    #     sns.kdeplot(low_p['Corrected Velocity'], label='0.2 psi', fill=True, color='C0')
    #     sns.kdeplot(old_low_p['Corrected Velocity'], label='old 0.2 psi', fill=True, color='C1')
    #     sns.kdeplot(med_p['Corrected Velocity'], label='0.8 psi', fill=True, color='C2')
    #     sns.kdeplot(old_med_p['Corrected Velocity'], label='old 0.8 psi', fill=True, color='C3')
    #     sns.kdeplot(high_p['Corrected Velocity'], label='1.2 psi', fill=True, color='C4')
    #     sns.kdeplot(old_high_p['Corrected Velocity'], label='old 1.2 psi', fill=True, color='C5')
    #     # sns.kdeplot(young_low_p['Corrected Velocity'], label='young 0.2 psi', fill=True)
    # plt.legend()
    # plt.title('Density Plot of different pressures young and old')
    # plt.show()

    # # Plot density old vs young
    # if hist:
    #     plt.hist(low_p['Corrected Velocity'], bins=20, label='0.2 psi', alpha = 0.5, density=density)
    #     plt.hist(old_low_p['Corrected Velocity'], bins=20, label='old 0.2 psi', alpha = 0.5, density=density)
    #     plt.hist(young_low_p['Corrected Velocity'], bins=20, label='young 0.2 psi', alpha = 0.5, density=density)
    #     plt.hist(med_p['Corrected Velocity'], bins=20, label='0.8 psi', alpha = 0.5, density=density)
    #     plt.hist(old_med_p['Corrected Velocity'], bins=20, label='old 0.8 psi', alpha = 0.5, density=density)
    #     plt.hist(young_med_p['Corrected Velocity'], bins=20, label='young 0.8 psi', alpha = 0.5, density=density)
    #     plt.hist(high_p['Corrected Velocity'], bins=20, label='1.2 psi', alpha = 0.5, density=density)
    #     plt.hist(old_high_p['Corrected Velocity'], bins=20, label='old 1.2 psi', alpha = 0.5, density=density)
    #     plt.hist(young_high_p['Corrected Velocity'], bins=20, label='young 1.2 psi', alpha = 0.5, density=density)
    # else:
    #     sns.kdeplot(low_p['Corrected Velocity'], label='0.2 psi', fill=True)
    #     sns.kdeplot(old_low_p['Corrected Velocity'], label='old 0.2 psi', fill=True)
    #     sns.kdeplot(young_low_p['Corrected Velocity'], label='young 0.2 psi', fill=True)
    #     sns.kdeplot(med_p['Corrected Velocity'], label='0.8 psi', fill=True)
    #     sns.kdeplot(old_med_p['Corrected Velocity'], label='old 0.8 psi', fill=True)
    #     sns.kdeplot(young_med_p['Corrected Velocity'], label='young 0.8 psi', fill=True)
    #     sns.kdeplot(high_p['Corrected Velocity'], label='1.2 psi', fill=True)
    #     sns.kdeplot(old_high_p['Corrected Velocity'], label='old 1.2 psi', fill=True)
    #     sns.kdeplot(young_high_p['Corrected Velocity'], label='young 1.2 psi', fill=True)
    # plt.legend()
    # plt.title('Density Plot of different pressures young and old')
    # plt.show()

    # Plot 0.2 psi old vs young
    if hist:
        plt.hist(low_p['Corrected Velocity'], bins=20, label='0.2 psi', alpha = 0.5, density=density)
        plt.hist(old_low_p['Corrected Velocity'], bins=20, label='old 0.2 psi', alpha = 0.5, density=density)
        plt.hist(young_low_p['Corrected Velocity'], bins=20, label='young 0.2 psi', alpha = 0.5, density=density)
    else:
        sns.kdeplot(low_p['Corrected Velocity'], label='0.2 psi', fill=True)
        sns.kdeplot(old_low_p['Corrected Velocity'], label='old 0.2 psi', fill=True)
        sns.kdeplot(young_low_p['Corrected Velocity'], label='young 0.2 psi', fill=True)
    plt.legend()
    plt.title('Density Plot of 0.2 psi young and old')
    plt.show()

    # Plot 0.8 psi old vs young
    if hist:
        plt.hist(med_p['Corrected Velocity'], bins=20, label='0.8 psi', alpha = 0.5, density=density)
        plt.hist(old_med_p['Corrected Velocity'], bins=20, label='old 0.8 psi', alpha = 0.5, density=density)
        plt.hist(young_med_p['Corrected Velocity'], bins=20, label='young 0.8 psi', alpha = 0.5, density=density)
    else:
        sns.kdeplot(med_p['Corrected Velocity'], label='0.8 psi', fill=True)
        sns.kdeplot(old_med_p['Corrected Velocity'], label='old 0.8 psi', fill=True)
        sns.kdeplot(young_med_p['Corrected Velocity'], label='young 0.8 psi', fill=True)
    plt.legend()
    plt.title('Density Plot of 0.8 psi young and old')
    plt.show()

    # Plot 1.2 psi old vs young
    if hist:
        plt.hist(high_p['Corrected Velocity'], bins=20, label='1.2 psi', alpha = 0.5, density=density)
        plt.hist(old_high_p['Corrected Velocity'], bins=20, label='old 1.2 psi', alpha = 0.5, density=density)
        plt.hist(young_high_p['Corrected Velocity'], bins=20, label='young 1.2 psi', alpha = 0.5, density=density)
    else:
        sns.kdeplot(high_p['Corrected Velocity'], label='1.2 psi', fill=True)
        sns.kdeplot(old_high_p['Corrected Velocity'], label='old 1.2 psi', fill=True)
        sns.kdeplot(young_high_p['Corrected Velocity'], label='young 1.2 psi', fill=True)
    plt.legend()
    plt.title('Density Plot of 1.2 psi young and old')
    plt.show()



    return 0

def plot_cdf_comp_pressure(df, title = 'CDF Plot for Different Pressures', write = False):
    """
    Plots the CDF for corrected velocities across different pressures and age groups within each pressure category.

    Parameters:
    - df: DataFrame containing 'Pressure', 'Corrected Velocity', and 'Age' columns.
    """
    pressures = df['Pressure'].unique()
    colors = ['C0', 'C1', 'C2']  # Colors for the plots
    labels = ['0.2 psi', '0.8 psi', '1.2 psi']  # Labels for different pressures

    # Plot CDF for different pressures
    plt.figure(figsize=(12, 8))
    for pressure, color, label in zip(pressures, colors, labels):
        subset = df[df['Pressure'] == pressure]
        values = np.sort(subset['Corrected Velocity'])
        cdf = np.arange(1, len(values) + 1) / len(values)
        plt.plot(values, cdf, label=f'{label}', color=color)

    plt.title(title)
    plt.xlabel('Velocity (um/s)')
    plt.ylabel('CDF')
    plt.legend()
    plt.grid(True)
    if write:
        filename = title.replace(' ', '_')
        filename += '.png'
        plt.savefig(filename, dpi=600)
        plt.close()
    else:
        plt.show()

    # Plot CDF for age groups within each pressure category
    plt.figure(figsize=(12, 8))
    for pressure, color, label in zip(pressures, colors, labels):
        for age_group, linestyle in [('old', 'dashed'), ('young', 'solid')]:
            subset = df[(df['Pressure'] == pressure) & (df['Age'] > 50 if age_group == 'old' else df['Age'] <= 50)]
            values = np.sort(subset['Corrected Velocity'])
            cdf = np.arange(1, len(values) + 1) / len(values)
            plt.plot(values, cdf, label=f'{age_group} {label}', color=color, linestyle=linestyle)

    plt.title(title + ' by Age Group')
    plt.xlabel('Velocity (um/s)')
    plt.ylabel('CDF')
    plt.legend()
    plt.grid(True)
    if write:
        filename = title.replace(' ', '_') + '_age_group'
        filename += '.png'
        plt.savefig(filename, dpi=600)
        plt.close()
    else:
        plt.show()
    return 0


def quantile_analysis(data, subset, quantiles=[0.25, 0.5, 0.75]):
    data_quantiles = np.quantile(data, quantiles)
    subset_quantiles = np.quantile(subset, quantiles)
    
    print("Quantile Analysis:")
    for q, dq, sq in zip(quantiles, data_quantiles, subset_quantiles):
        print(f"{int(q*100)}th percentile - Entire Dataset: {dq}, Subset: {sq}")
    return 0

def calc_norm_cdfs(data):
    cdfs = []
    for participant in data:
        participant_sorted = np.sort(participant)
        p_participant = 1. * np.arange(len(participant)) / (len(participant) - 1)
        cdf = np.vstack([participant_sorted, p_participant])
        cdfs.append(cdf)
    cdfs = np.array(cdfs)
    cdfs = np.mean(cdfs, axis=0)
    return cdfs

def calculate_cdf(data, normalize=False):
    if normalize:
        return calc_norm_cdfs(data)
    else:
        sorted_data = np.sort(data)
        p = np.linspace(0, 1, len(data))
        return sorted_data, p
    
def plot_cdf(data, subsets, labels=['Entire Dataset', 'Subset'], title='CDF Comparison', 
             write=False, normalize=False, variable = 'Age'):
    """
    Plots the CDF of the entire dataset and the inputted subsets.

    Args:
        data (array-like): The entire dataset
        subsets (list of array-like): The subsets to be compared
        labels (list of str): The labels for the entire dataset and the subsets
        title (str): The title of the plot
        write (bool): Whether to write the plot to a file
        normalize (bool): Whether to normalize the CDF
    
    Returns:
        0 if successful, 1 if no subsets provided
    """
    # Set up style and font
    sns.set_style("whitegrid")
    source_sans = FontProperties(fname='C:\\Users\\gt8ma\\Downloads\\Source_Sans_3\\static\\SourceSans3-Regular.ttf')
    
    plt.rcParams.update({
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'font.size': 7, 'axes.labelsize': 7,
        'xtick.labelsize': 6, 'ytick.labelsize': 6,
        'legend.fontsize': 5, 'lines.linewidth': 0.5
    })

    # Set color palette based on variable
    if variable == 'Age':
        base_color = '#1f77b4'##3FCA54''BDE4A7 #A1E5AB
    elif variable == 'SYS_BP':
        base_color = '2ca02c'#80C6C3 #ff7f0e
    elif variable == 'Sex':
        base_color = '674F92'#947EB0#2ca02c#CAC0D89467bd
    else:
        raise ValueError(f"Unsupported variable: {variable}")

    palette = create_monochromatic_palette(base_color)
    # palette = adjust_saturation_of_colors(palette, saturation_scale=1.3)
    palette = adjust_brightness_of_colors(palette, brightness_scale=.2)
    sns.set_palette(palette)

    if not subsets:
        return 1

    fig, ax = plt.subplots(figsize=(2.4, 2.0))

    # Plot main dataset
    x, y = calculate_cdf(data, normalize)
    ax.plot(x, y, label=labels[0])

    # Plot subsets
    for i in  range(len(subsets)):
        if i == 0:
            i_color = 0
            dot_color = 0
        elif i == 1:
            i_color = 3
            dot_color = 2
        elif i == 2:
            i_color = 4
            dot_color=3
        x, y = calculate_cdf(subsets[i], normalize)
        ax.plot(x, y, label=labels[i+1], linestyle='--', color=palette[i_color])

    ax.set_ylabel('CDF')
    ax.set_xlabel('Velocity (um/s)' if 'Pressure' not in title else 'Pressure (psi)')
    ax.set_title(title, fontsize=8)
    
    # Adjust legend
    ax.legend(loc='lower right', bbox_to_anchor=(1, 0.2))
    
    ax.grid(True, linewidth=0.3)

    # Adjust layout to prevent cutting off labels
    plt.tight_layout()

    if write:
        save_plot(fig, title, dpi=300)
    else:
        plt.show()

    return 0

def save_plot(fig, title, dpi=600):
    filename = f"{title.replace(' ', '_')}.png"
    filepath = os.path.join('C:\\Users\\gt8ma\\capillary-flow\\results', filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
def plot_individual_cdfs(data, title='CDF Comparison', write=False):
    """
    Plots the CDF for each individual in the data, colored by age.

    Args:
        data (DataFrame): The entire dataset with columns 'Participant', 'Age', and the data values.
        title (str): The title of the plot.
        write (bool): Whether to write the plot to a file.

    Returns:
        None
    """
    # Get unique participants
    participants = data['Participant'].unique()

    # Setup plot
    plt.figure(figsize=(10, 6))

    for participant in participants:
        participant_data = data[data['Participant'] == participant]
        participant_age = participant_data['Age'].iloc[0]
        values = participant_data['Corrected Velocity']  # Assuming the column with data is named 'Value'
        values_sorted = np.sort(values)
        p = 1. * np.arange(len(values)) / (len(values) - 1)
        linestyle = '--' if participant_age > 50 else '-'
        color = plt.cm.viridis((participant_age - data['Age'].min()) / (data['Age'].max() - data['Age'].min()))
        plt.plot(values_sorted, p, label=f'Participant {participant} (Age {participant_age})',
                 linestyle=linestyle, color=color)

    plt.ylabel('CDF')
    plt.xlabel('Velocity (um/s)')  # Adjust the label to match the actual data
    plt.title(title)
    plt.legend()
    plt.grid(True)

    # Save or show plot
    if write:
        filename = title.replace(' ', '_') + '.png'
        plt.savefig(filename, dpi=600)
        plt.close()
    else:
        plt.show()
    return 0

def plot_boxplot(data, subset, labels=['Entire Dataset', 'Subset']):
    # Combine data and subset into a single dataset for plotting
    combined_data = np.concatenate([data, subset])
    # Create a list of labels corresponding to the data and subset
    combined_labels = np.concatenate([[labels[0]]*len(data), [labels[1]]*len(subset)])
    
    # Create a DataFrame for easier plotting with seaborn
    import pandas as pd
    df = pd.DataFrame({'Value': combined_data, 'Group': combined_labels})
    
    # Plotting
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Group', y='Value', data=df)
    plt.title('Box Plot of Entire Dataset vs. Subset')
    plt.show()
    return 0

def plot_violinplot(data, subset, labels=['Entire Dataset', 'Subset']):
    # Combine data and subset into a single dataset for plotting
    combined_data = np.concatenate([data, subset])
    # Create a list of labels corresponding to the data and subset
    combined_labels = np.concatenate([[labels[0]]*len(data), [labels[1]]*len(subset)])
    
    # Create a DataFrame for easier plotting with seaborn
    import pandas as pd
    df = pd.DataFrame({'Value': combined_data, 'Group': combined_labels})
    
    # Plotting
    plt.figure(figsize=(8, 6))
    sns.violinplot(x='Group', y='Value', data=df, inner='quartile')
    plt.title('Violin Plot of Entire Dataset vs. Subset')
    plt.show()
    return 0

def plot_box_and_whisker(df_entire_data, df_subset1, df_subset2, column, variable = 'Age', log_scale = False):
    # Extract the column of interest from each dataframe
    data_entire = df_entire_data[column].rename('Entire Dataset')
    if variable == 'Age':
        data_subset1 = df_subset1[column].rename('Old')
        data_subset2 = df_subset2[column].rename('Young')
    else:
        data_subset1 = df_subset1[column].rename('High BP')
        data_subset2 = df_subset2[column].rename('Normal BP')
    

    # Combine the data into a single dataframe for plotting
    combined_data = pd.concat([data_entire, data_subset1, data_subset2], axis=1)
    
    
    # Plot the box and whisker plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=combined_data, showfliers=False)
    if log_scale:
        plt.yscale('log')
    if variable == 'Age':
        plt.title('Comparison of Velocities by Age Group')
    else:
        plt.title('Comparison of Velocities by Blood Pressure Group')
    plt.ylabel('Velocity (um/s)')
    plt.show()
    return 0

def plot_violin(df_entire_data, df_subset1, df_subset2, column, log_scale = False):
    # Extract the column of interest and create a category column
    data_entire = df_entire_data.assign(Category='Entire Dataset')[[column, 'Category']]
    data_subset1 = df_subset1.assign(Category='Old')[[column, 'Category']]
    data_subset2 = df_subset2.assign(Category='Young')[[column, 'Category']]

    # Combine the data into a single dataframe for plotting
    combined_data = pd.concat([data_entire, data_subset1, data_subset2])

    if log_scale:
        # Apply logarithmic transformation to the data
        combined_data[column] = combined_data[column].apply(lambda x: np.log(x) if x > 0 else None)

    # Plot the violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Category', y=column, data=combined_data)
    plt.title('Comparison of Log Velocities by Age Group' if log_scale else 'Comparison of Velocities by Age Group')
    plt.ylabel('Log Velocity' if log_scale else 'Velocity (um/s)')
    plt.show()
    return 0

def collapse_df(df):
    # Calculate median velocity for specific pressures and overall median velocity
    pressure_medians = df.groupby(['Participant', 'Pressure'])['Corrected Velocity'].median().unstack()
    pressure_medians.columns = [f'Pressure {col}' for col in pressure_medians.columns]

    # Overall median velocity and median SYS_BP
    overall_median_velocity = df.groupby('Participant')['Corrected Velocity'].median().rename('Median Velocity')
    median_sys_bp = df.groupby('Participant')['SYS_BP'].median().rename('Median SYS_BP')

    # Combine all the data
    final_df = pd.concat([pressure_medians, overall_median_velocity, median_sys_bp], axis=1).reset_index()

    # Assume the Age for each participant is constant and just take the first one.
    ages = df.groupby('Participant')['Age'].first()

    # Combine age into the final DataFrame
    final_df = pd.merge(final_df, ages, left_on='Participant', right_index=True)
    
    # If there is an 'Area Score' column, add it to the final DataFrame
    if 'Area Score' in df.columns:
        area_scores = df.groupby('Participant')['Area Score'].mean().rename('Area Score')
        final_df = pd.merge(final_df, area_scores, left_on='Participant', right_index=True)
    
    # If there is a 'Log Area Score' column, add it to the final DataFrame
    if 'Log Area Score' in df.columns:
        log_area_scores = df.groupby('Participant')['Log Area Score'].mean().rename('Log Area Score')
        final_df = pd.merge(final_df, log_area_scores, left_on='Participant', right_index=True)

    # If there is a 'KS Statistic' column, add it to the final DataFrame
    if 'KS Statistic' in df.columns:
        ks_stats = df.groupby('Participant')['KS Statistic'].mean().rename('KS Statistic')
        final_df = pd.merge(final_df, ks_stats, left_on='Participant', right_index=True)

    # if there is a 'EMD Score' column, add it to the final DataFrame
    if 'EMD Score' in df.columns:
        emd_scores = df.groupby('Participant')['EMD Score'].mean().rename('EMD Score')
        final_df = pd.merge(final_df, emd_scores, left_on='Participant', right_index=True)
    
    # if there is a 'Sex' column, add it to the final DataFrame
    if 'Sex' in df.columns:
        # Make sex 0 for males and 1 for women
        sex = df.groupby('Participant')['Sex'].first().map({'M': 0, 'F': 1})
        final_df = pd.merge(final_df, sex, left_on='Participant', right_index=True)

    return final_df

def plot_stats(df):
    # Setting the style
    sns.set_theme(style="whitegrid")

    # Plotting distributions of Median Velocity, Age, and Median SYS_BP. 
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    sns.histplot(df['Median Velocity'], kde=True, ax=ax[0], color='skyblue')
    ax[0].set_title('Distribution of Median Velocity')

    sns.histplot(df['Age'], kde=True, ax=ax[1], color='lightgreen')
    ax[1].set_title('Distribution of Age')

    sns.histplot(df['Median SYS_BP'], kde=True, ax=ax[2], color='salmon')
    ax[2].set_title('Distribution of Median SYS_BP')

    plt.tight_layout()
    plt.show()

    # Pair plot to visualize relationships between Median Velocity and other features
    sns.pairplot(df[['Median Velocity', 'Pressure 0.2', 'Pressure 0.8', 'Pressure 1.2', 'Age', 'Median SYS_BP', 'Area Score', 'Log Area Score', 'KS Statistic', 'EMD Score']])
    plt.show()

    # Correlation matrix
    corr_matrix = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()
    return 0

def make_models(df, variable = 'Median Velocity', log = False, plot = False):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.model_selection import RandomizedSearchCV

    # preparing the data
    if log:
        X = df.drop(['Participant', 'Median Velocity', 'Area Score', 'Log Area Score', 'EMD Score', 'KS Statistic',                      
                     'Pressure 0.2', 'Pressure 0.4', 'Pressure 0.6', 'Pressure 0.8', 'Pressure 1.0', 'Pressure 1.2'], axis=1)  # Using log pressures, age, and sys_bp as features
    else: 
        X = df.drop(['Participant', 'Log Median Velocity', 'Area Score', 'Log Area Score', 'EMD Score', 'KS Statistic',
                     'Log Pressure 0.2', 'Log Pressure 0.4', 'Log Pressure 0.6', 'Log Pressure 0.8', 'Log Pressure 1.0', 'Log Pressure 1.2'], axis=1)  # Using pressures, age, and sys_bp as features
    
    if variable == 'Median Velocity':
        Y = df['Median Velocity']
        X.drop(['Median Velocity'], axis=1, inplace=True)
    elif variable == 'Log Median Velocity':
        Y = df['Log Median Velocity']
        X.drop(['Log Median Velocity'], axis=1, inplace=True)
    elif variable == 'Log Area Score':
        Y = df['Log Area Score']
    elif variable == 'Area Score':
        Y = df['Area Score']
    else:
        Y = df['Median Velocity']
        X.drop(['Log Median Velocity', 'Median Velocity'], axis=1, inplace=True)


    print(X.columns, Y.name)

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Initializing models
    linear_reg = LinearRegression()
    random_forest = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # # Optimized Random Forest caused issues with residuals
    # random_forest_optimized = RandomForestRegressor(n_estimators=200, min_samples_split=2, min_samples_leaf=1, random_state=42)
    # random_forest = random_forest_optimized

    # Training Linear Regression
    linear_reg.fit(X_train, y_train)
    y_pred_lr = linear_reg.predict(X_test)

    # Training Random Forest Regressor
    random_forest.fit(X_train, y_train)
    y_pred_rf = random_forest.predict(X_test)

    # Evaluating models
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    rmse_lr = mean_squared_error(y_test, y_pred_lr, squared=False)

    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)

    print(f'Linear Regression MAE: {mae_lr}, RMSE: {rmse_lr}')
    print(f'Random Forest MAE: {mae_rf}, RMSE: {rmse_rf}')

    if plot:
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))

        # Actual vs. Predicted for Linear Regression
        axs[0, 0].scatter(y_test, y_pred_lr, color='blue', alpha=0.5)
        axs[0, 0].plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=2)
        axs[0, 0].set_title('Linear Regression: Actual vs. Predicted')
        axs[0, 0].set_xlabel('Actual')
        axs[0, 0].set_ylabel('Predicted')

        # Residuals for Linear Regression
        residuals_lr = y_test - y_pred_lr
        axs[1, 0].scatter(y_pred_lr, residuals_lr, color='red', alpha=0.5)
        axs[1, 0].hlines(y=0, xmin=y_pred_lr.min(), xmax=y_pred_lr.max(), colors='k', lw=2)
        axs[1, 0].set_title('Linear Regression: Residuals')
        axs[1, 0].set_xlabel('Predicted')
        axs[1, 0].set_ylabel('Residuals')

        # Actual vs. Predicted for Random Forest
        axs[0, 1].scatter(y_test, y_pred_rf, color='green', alpha=0.5)
        axs[0, 1].plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=2)
        axs[0, 1].set_title('Random Forest: Actual vs. Predicted')
        axs[0, 1].set_xlabel('Actual')
        axs[0, 1].set_ylabel('Predicted')

        # Residuals for Random Forest
        residuals_rf = y_test - y_pred_rf
        axs[1, 1].scatter(y_pred_rf, residuals_rf, color='orange', alpha=0.5)
        axs[1, 1].hlines(y=0, xmin=y_pred_rf.min(), xmax=y_pred_rf.max(), colors='k', lw=2)
        axs[1, 1].set_title('Random Forest: Residuals')
        axs[1, 1].set_xlabel('Predicted')
        axs[1, 1].set_ylabel('Residuals')

        plt.tight_layout()
        plt.show()

    # # Randomized Search CV for hyperparameter tuning
    # # Simplifying the parameter grid and reducing the number of iterations for demonstration
    # simplified_param_dist = {
    #     'n_estimators': [100, 200],
    #     'max_depth': [None, 10],
    #     'min_samples_split': [2, 5],
    #     'min_samples_leaf': [1, 2]
    # }

    # # Initializing the Randomized Search CV object with simplified parameters
    # simplified_random_search = RandomizedSearchCV(estimator=random_forest, param_distributions=simplified_param_dist, n_iter=10, cv=5, verbose=2, random_state=42, n_jobs=-1, scoring='neg_mean_absolute_error')

    # # Fitting it to the data
    # simplified_random_search.fit(X_train, y_train)

    # # Best parameters and score from the simplified search
    # simplified_best_params = simplified_random_search.best_params_
    # simplified_best_score = -simplified_random_search.best_score_  # Note: scores are negative in sklearn

    # print(simplified_best_params, simplified_best_score)
    return mae_lr, rmse_lr, mae_rf, rmse_rf

def make_log_df(df, plot = False):
    # Log-transforming selected features and the target variable
    df['Log Pressure 0.2'] = np.log1p(df['Pressure 0.2'])
    df['Log Pressure 0.4'] = np.log1p(df['Pressure 0.4'])
    df['Log Pressure 0.6'] = np.log1p(df['Pressure 0.6'])
    df['Log Pressure 0.8'] = np.log1p(df['Pressure 0.8'])
    df['Log Pressure 1.0'] = np.log1p(df['Pressure 1.0'])
    df['Log Pressure 1.2'] = np.log1p(df['Pressure 1.2'])
    df['Log Median Velocity'] = np.log1p(df['Median Velocity'])

    if plot:
        # Plot log-transformed features vs Age
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        sns.scatterplot(y='log_Pressure_0.2', x='Age', data=df, ax=axs[0])
        axs[0].set_title('Log-transformed Pressure 0.2 vs. Age')
        sns.scatterplot(y='log_Pressure_0.8', x='Age', data=df, ax=axs[1])
        axs[1].set_title('Log-transformed Pressure 0.8 vs. Age')
        sns.scatterplot(y='log_Pressure_1.2', x='Age', data=df, ax=axs[2])
        axs[2].set_title('Log-transformed Pressure 1.2 vs. Age')
        plt.tight_layout()
        plt.show()

        # Plot log-transformed features vs SYS_BP
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        sns.scatterplot(y='log_Pressure_0.2', x='Median SYS_BP', data=df, ax=axs[0])
        axs[0].set_title('Log-transformed Pressure 0.2 vs. Median SYS_BP')
        sns.scatterplot(y='log_Pressure_0.8', x='Median SYS_BP', data=df, ax=axs[1])
        axs[1].set_title('Log-transformed Pressure 0.8 vs. Median SYS_BP')
        sns.scatterplot(y='log_Pressure_1.2', x='Median SYS_BP', data=df, ax=axs[2])
        axs[2].set_title('Log-transformed Pressure 1.2 vs. Median SYS_BP')
        plt.tight_layout()
        plt.show()

        # Plot log-transformed Median Velocity vs Age
        plt.figure(figsize=(8, 5))
        sns.scatterplot(y='log_Median_Velocity', x='Age', data=df)
        plt.title('Log-transformed Median Velocity vs. Age')
        plt.show()
    return df

def calculate_video_median_velocity(df, dimensionless= False):
    # Make a copy of the DataFrame to avoid modifying the original one
    df_copy = df.copy()

    if dimensionless:
        video_median_velocity = df_copy.groupby(['Participant', 'Video'])['Dimensionless Velocity'].median().reset_index(name='Video Median Dimensionless Velocity')
        merged_df = pd.merge(df_copy, video_median_velocity, on=['Participant', 'Video'], how='left')
        collapsed_df = merged_df.drop_duplicates(subset=['Participant', 'Video'])
    else:
        # Group by 'participant' and 'video', then calculate the median of 'Corrected Velocity'
        video_median_velocity = df_copy.groupby(['Participant', 'Video'])['Corrected Velocity'].median().reset_index(name='Video Median Velocity')

        # Merge the median values back to the original DataFrame copy
        merged_df = pd.merge(df_copy, video_median_velocity, on=['Participant', 'Video'], how='left')

        # Drop duplicates to get a collapsed DataFrame with unique participant-video pairs
        collapsed_df = merged_df.drop_duplicates(subset=['Participant', 'Video'])

    return collapsed_df

def plot_median_velocity_of_videos(df):
    # Ensure the DataFrame is sorted by participant and then by video to get the correct order
    sorted_df = df.sort_values(by=['Participant', 'Location', 'Video'])

    # Set the plotting style for better readability
    plt.style.use('seaborn-darkgrid')

    # Create a line plot for each participant, ordered by video number but plotted against Pressure
    for participant, group in sorted_df.groupby('Participant'):
        for location, location_group in group.groupby('Location'):
            # Sort the group by Pressure to get the x-axis values, but trace the line by video order
            group_sorted_by_video = location_group.sort_values(by='Video')
            plt.plot(group_sorted_by_video['Pressure'], group_sorted_by_video['Video Median Velocity'], marker='o', label=f'Participant {participant}')

            plt.xlabel('Pressure')
            plt.ylabel('Video Median Velocity')
            plt.title(f'Video Median Velocity by Pressure for Participant {participant} at Location {location}')
            plt.legend()
            plt.show()
    return 0


def compare_log_and_linear(df, variable = 'Median Velocity', plot = False):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.ensemble import RandomForestRegressor

    # Assuming `df` is your DataFrame
    df = make_log_df(df)

    # Splitting both datasets into training and testing sets for all pressure columns
    features_original = ['Pressure 0.2', 'Pressure 0.8', 'Pressure 1.2']  # Example original features
    features_transformed = ['log_Pressure_0.2', 'log_Pressure_0.8', 'log_Pressure_1.2']  # Corresponding log-transformed features
    
    X_original = df[features_original]
    X_transformed = df[features_transformed]
    if variable == 'Median Velocity':
        y_original = df['Median Velocity']
        y_transformed = df['log_Median_Velocity']
    elif variable == 'Area Score':
        y_original = df['Area Score']
        y_transformed = df['Log Area Score']
    else:
        y_original = df['Age']
        y_transformed = df['Age']
    
    X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X_original, y_original, test_size=0.2, random_state=42)
    X_train_trans, X_test_trans, y_train_trans, y_test_trans = train_test_split(X_transformed, y_transformed, test_size=0.2, random_state=42)
    
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    for name, model in models.items():
        # Original data
        model.fit(X_train_orig, y_train_orig)
        predictions_orig = model.predict(X_test_orig)
        
        # Log-transformed data
        model.fit(X_train_trans, y_train_trans)
        predictions_trans = model.predict(X_test_trans)
        
        # Evaluation
        mse_original = mean_squared_error(y_test_orig, predictions_orig)
        r2_original = r2_score(y_test_orig, predictions_orig)
        
        mse_transformed = mean_squared_error(np.expm1(y_test_trans), np.expm1(predictions_trans))
        r2_transformed = r2_score(np.expm1(y_test_trans), np.expm1(predictions_trans))
        
        print(f"{name} - Original Data - MSE: {mse_original:.4f}, R2: {r2_original:.4f}")
        print(f"{name} - Log-transformed Data - MSE: {mse_transformed:.4f}, R2: {r2_transformed:.4f}")
        
        if plot:
            # Actual vs. Predicted Plot
            plt.figure(figsize=(14, 6))
            plt.subplot(1, 2, 1)
            sns.scatterplot(x=y_test_orig, y=predictions_orig).set(title=f'{name} - Original Data: Actual vs. Predicted variable {variable}')
            plt.subplot(1, 2, 2)
            sns.scatterplot(x=np.expm1(y_test_trans), y=np.expm1(predictions_trans)).set(title=f'{name} - Log-transformed Data: Actual vs. Predicted variable {variable}')
            plt.show()
            
            # Residual Plot
            plt.figure(figsize=(14, 6))
            plt.subplot(1, 2, 1)
            sns.residplot(x=predictions_orig, y=y_test_orig, lowess=True).set(title=f'{name} - Original Data: Residuals variable {variable}')
            plt.subplot(1, 2, 2)
            sns.residplot(x=np.expm1(predictions_trans), y=np.expm1(y_test_trans), lowess=True).set(title=f'{name} - Log-transformed Data: Residuals variable {variable}')
            plt.show()
    
    return 0

def get_bad_apples(summary_df):
    # Filter the dataframe where 'Correct', 'Zero', and 'Correct2' are all False
    filtered_df = summary_df[(summary_df['Correct'] == 'f') & 
                             (summary_df['Zero'] == 'f') ] # &       (summary_df['Correct2'] == 'f')
    return filtered_df

def age_bin_accuracy(y_true, y_pred, threshold=50):
    """
    Custom accuracy scorer that categorizes predicted and true ages into bins based on a threshold.
    
    Parameters:
    - y_true: array-like of true ages.
    - y_pred: array-like of predicted ages.
    - threshold: age threshold for categorizing into bins (default is 50).
    
    Returns:
    - Accuracy of the bin categorization.
    """
    # Convert ages to binary categories (0: young, 1: old)
    y_true_bin = (y_true >= threshold).astype(int)
    y_pred_bin = (y_pred >= threshold).astype(int)
    
    # Calculate accuracy
    return accuracy_score(y_true_bin, y_pred_bin)

def perform_logistic_regression(dataframe, features, target, cv_folds=5, score_func='banana'):
    """
    Fits a logistic regression model and evaluates its performance.

    Args:
    - dataframe: The pandas DataFrame containing the data.
    - features: List of column names to use as features.
    - target: The name of the target column.
    - cv_folds: Number of cross-validation folds to use.

    Returns:
    - model: The fitted LogisticRegression model.
    """
    X = dataframe[features]
    y = dataframe[target]
    # # set y values to 0 or 1 (0 for Age < 50, 1 for Age >= 50)
    # y = y >= 50


    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    # Creating and fitting the logistic regression model within a pipeline
    model = make_pipeline(StandardScaler(), LogisticRegression())
    model.fit(X_train, y_train)

    # Making predictions and evaluating the model
    y_pred = model.predict(X_test)
    # print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    # print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print('This is for perform logistic regression')
    print(f'y_pred: {y_pred}')
    print(f'y_test: {y_test}')

    # Now add cross-validation
    if score_func is None:
        score_func = 'accuracy'
    else:
        custom_accuracy_scorer = make_scorer(age_bin_accuracy, greater_is_better=True)
        score_func = custom_accuracy_scorer
    accuracy_scores = cross_val_score(model, X, y, cv=cv_folds, scoring=score_func)
    average_accuracy = np.mean(accuracy_scores)
    print(f"Average accuracy with {cv_folds}-fold cross-validation: {average_accuracy}")
    print(features)

    # calculate Recall, Precision, F1
    y_test_bool = y_test >= 50
    y_pred_bool = y_pred >= 50
    # turn True and False into 1 and 0
    y_test_bool = y_test_bool.astype(int)
    y_pred_bool = y_pred_bool.astype(int)
    print(f'y_test_bool: {y_test_bool}')
    print(f'y_pred_bool: {y_pred_bool}')

    # Calculate Precision, Recall, F1
    precision = precision_score(y_test_bool, y_pred_bool, average='binary')
    recall = recall_score(y_test_bool, y_pred_bool, average='binary')
    f1 = f1_score(y_test_bool, y_pred_bool, average='binary')

    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1: {f1}')
    
    print('End of perform logistic regression')
    return model

def predict_age_with_linear_regression(dataframe, features):
    """
    Fits a linear regression model to predict age.

    Args:
    - dataframe: The pandas DataFrame containing the data.
    - features: List of column names to use as features.

    Returns:
    - age_predictions: Predicted ages for the test set.
    """
    X = dataframe[features]
    y = dataframe['Age']  # Assuming 'Age' is the column with age values

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    # Creating and fitting the linear regression model within a pipeline
    model = make_pipeline(StandardScaler(), LinearRegression())
    age_predictions_CV = cross_val_predict(model, X, y, cv=5)
    model.fit(X_train, y_train)

    # Making predictions on the test set
    age_predictions = model.predict(X_test)
    age_buckets = np.where(age_predictions >= 50, 1, 0)
    age_buckets_CV = np.where(age_predictions_CV >= 50, 1, 0)

    # You could then evaluate the classification using actual age buckets in y_test
    y_test_buckets = np.where(y_test >= 50, 1, 0)
    y_test_buckets_CV = np.where(y >= 50, 1, 0)
    precision = precision_score(y_test_buckets, age_buckets)
    recall = recall_score(y_test_buckets, age_buckets)
    f1 = f1_score(y_test_buckets, age_buckets)

    # plot confusion matrix
    plot_confusion_matrix(y_test_buckets, age_buckets, features = features, threshold=.5, class_names=['Young', 'Old'])
    print(f"Accuracy: {accuracy_score(y_test_buckets, age_buckets)}")

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    precision_CV = precision_score(y_test_buckets_CV, age_buckets_CV)
    recall_CV = recall_score(y_test_buckets_CV, age_buckets_CV)
    f1_CV = f1_score(y_test_buckets_CV, age_buckets_CV)

    print(f"Precision CV: {precision_CV:.4f}")
    print(f"Recall CV: {recall_CV:.4f}")
    print(f"F1 Score CV: {f1_CV:.4f}")

    # plot confusion matrix
    plot_confusion_matrix(y_test_buckets_CV, age_buckets_CV, features = features, threshold=.5, class_names=['Young CV', 'Old CV'])
    print(f"Accuracy CV: {accuracy_score(y_test_buckets_CV, age_buckets_CV)}")

    print('End of predict age with linear regression')

    
    return age_predictions, y_test



def perform_lasso_regression(dataframe, features, target):
    """
    Fits a Lasso regression model with cross-validation and prints feature coefficients.

    Parameters:
    - dataframe: The pandas DataFrame containing the data.
    - features: List of column names to use as features.
    - target: The name of the target column.

    Returns:
    - model: The fitted LassoCV model.
    """
    X = dataframe[features]
    y = dataframe[target]

    # Creating and fitting the Lasso model within a pipeline
    model = make_pipeline(StandardScaler(), LassoCV(cv=5, max_iter=5000))
    model.fit(X, y)

    # Accessing the Lasso model directly to get the coefficients
    lasso = model.named_steps['lassocv']
    coefficients = pd.Series(lasso.coef_, index=features)

    # Printing the coefficients
    print('Lasso Regression Model:')
    print("Feature coefficients:")
    print(coefficients)

    print('End of perform lasso regression')

    return model

def perform_lasso_regression_and_evaluate(dataframe, features, target):
    X = dataframe[features]
    y = dataframe[target]

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    # Creating and fitting the Lasso model within a pipeline
    model = make_pipeline(StandardScaler(), LassoCV(cv=5))
    model.fit(X_train, y_train)

    # Predicting on the test set
    y_pred = model.predict(X_test)

    print(f'This is for perform lasso regression and evaluate')

    print(f'y_pred: {y_pred} lasso')
    print(f'y_test: {y_test} lasso')

    # Evaluation for regression
    print("R-squared:", r2_score(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))

    # Check number correctly classified: 50 year threshold
    y_pred = y_pred >= 50
    y_test = y_test >= 50

    print(confusion_matrix(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))

    # # select features which are not zero: 
    # lasso = model.named_steps['lassocv']
    # print(lasso.coef_.shape)
    # print(np.asarray(features).shape)
    # coefficients = pd.DataFrame(features, lasso.coef_, columns=['Feature', 'Coefficient'])
    # print("Banana:")
    # print(coefficients) 
    # for coefficient in coefficients:
    #     print(coefficient)
    #     print('asdf;lkjasdfl;jkasdf;klj')
    # # the second column of coefficients is the value of the coefficients
    brute_force_features = f"Lasso of ['Pressure 0.2', 'Pressure 1.2', 'Median SYS_BP', 'Log Pressure 0.4']"
    plot_confusion_matrix(y_test, y_pred, features = brute_force_features, threshold=.5, class_names=['Young Lasso', 'Old Lasso'])

    print('End of perform lasso regression and evaluate')


    return model

def perform_random_forest_classification(dataframe, features, target):
    """
    Fits a Random Forest classifier to the data and evaluates its performance.

    Parameters:
    - dataframe: The pandas DataFrame containing the data.
    - features: List of column names to use as features.
    - target: The name of the target column.

    Returns:
    - model: The fitted RandomForestClassifier model.
    """
    X = dataframe[features]
    y = dataframe[target]

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # It's often a good idea to scale your data for other models, but Random Forest does not require it as it is not sensitive to the variance in the data.
    # Creating and fitting the Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predicting on the test set
    y_pred = model.predict(X_test)

    # Evaluating the model
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Plotting feature importances
    feature_importances = pd.Series(model.feature_importances_, index=features)
    feature_importances.nlargest(len(features)).plot(kind='barh')
    plt.title('Feature Importances')
    plt.show()

    return model

def calculate_auc(y_true, y_scores, features = None, plot=False):
    # Assuming y_scores are the continuous outputs from the Lasso model
    # Binarize predictions based on a threshold if necessary
    # For example, threshold = 0.5, y_pred = [1 if y > threshold else 0 for y in y_scores]

    # Calculate AUC
    auc = roc_auc_score(y_true, y_scores)

    if plot:
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
        plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        if features:
            plt.title(f'Receiver Operating Characteristic (ROC) for \n{features}')
        else:
            plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.show()

    return auc

def plot_confusion_matrix(y_true, y_scores, features = None, threshold=0.5, class_names=None):
    """
    Plots a confusion matrix using actual labels and predicted scores.

    Parameters:
    - y_true: Array-like of true labels.
    - y_scores: Array-like of scores predicted by the model.
    - threshold: Threshold for converting scores to binary predictions.
    - class_names: List of class names for the plot. For binary classification, use ['Negative', 'Positive'].
    """
    # Convert scores to binary predictions
    y_pred = np.where(y_scores > threshold, 1, 0)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    # add list of features to the title
    if features:
        plt.title(f'Confusion Matrix for \n{features}')
    else:
        plt.title('Confusion Matrix')
    plt.show()

def make_roc_curve_one_var(df, feature, target='Age', flip = False):
    if target != 'Age':
        raise ValueError('Please choose target for this function (only Age is supported)')
    age_threshold = 50
    if flip:
        # Categorize 'old' (0) and 'young' (1) based on age threshold
        df['Age Category'] = (df['Age'] < age_threshold).astype(int)
    else:
        # Categorize 'old' (1) and 'young' (0) based on age threshold
        df['Age Category'] = (df['Age'] >= age_threshold).astype(int)

    # Assuming a simple linear relationship for illustration
    # Adjust 'LogArea Score' threshold for classification
    # Normally, you'd use a model or some logic here
    thresholds = np.linspace(df[feature].min(), df[feature].max(), 50)
    tprs = []
    fprs = []

    for threshold in thresholds:
        df['Predicted'] = (df[feature] >= threshold).astype(int)
        fpr, tpr, _ = roc_curve(df['Age Category'], df['Predicted'])
        tprs.append(tpr[1])
        fprs.append(fpr[1])

    # Plot ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fprs, tprs, marker='o', linestyle='-', color='blue')
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Age Classification using ' + feature)

    # Calculate AUC
    roc_auc = auc(fprs, tprs)

    # Print AUC on the plot
    plt.text(0.6, 0.2, f'AUC: {roc_auc:.2f}', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    plt.show()

def run_regression(df, plot = False):
    """
    Runs a linear regression analysis on the inputted DataFrame and plots results.

    Args:
        df (DataFrame): The DataFrame to be analyzed
    
    Returns:
        0 if successful
    """
    collapsed_df = collapse_df(df)
    collapsed_df = make_log_df(collapsed_df)
    if plot:
        plot_stats(collapsed_df)

    # make_models(collapsed_df, variable = 'Area Score', log = False, plot=False)
    # make_models(collapsed_df, variable = 'Log Area Score', log = True, plot=False)
    # make_models(collapsed_df, variable = 'Median Velocity', log = False, plot=False)
        
    # compare_log_and_linear(collapsed_df, "Area Score", plot=False)
    logistic_regression_features = ['Log Area Score', 'Pressure 0.2', 'Pressure 0.4', 'Pressure 1.2', 
                                    'Median SYS_BP']
    # make lasso features all features not including the target Age
    lasso_features = collapsed_df.columns.tolist()
    # lasso_features = ['Age', 'Participant', 'Log Area Score', 'Pressure 0.2', 'Pressure 1.2', 'Median SYS_BP']
    lasso_features.remove('Participant')
    lasso_features.remove('Age')
    # remove_features = ['Log Pressure 0.2', 'Log Pressure 0.4', 'Log Pressure 0.6', 'Log Pressure 0.8', 'Log Pressure 1.0', 'Log Pressure 1.2']
    # lasso_features = [feature for feature in lasso_features if feature not in remove_features]
    target = 'Age'

    # # Make models
    # logistic_model = perform_logistic_regression(collapsed_df, logistic_regression_features, target)
    lasso_model = perform_lasso_regression(collapsed_df, lasso_features, target)
    print(lasso_model)
    lasso_model_eval = perform_lasso_regression_and_evaluate(collapsed_df, lasso_features, target)
    print(lasso_model_eval)

    # # # Calculate AUC with age threshold of 50
    # y_true = (collapsed_df['Age'] > 50).astype(int)
    # y_scores = lasso_model_eval.predict(collapsed_df[lasso_features])
    # auc = calculate_auc(y_true, y_scores, plot=True)

    # # # plot confusion matrix for lasso model
    # plot_confusion_matrix(y_true, y_scores, threshold=50, class_names=['Under 50', 'Over 50'])   

    # -------------------------------------------------------------------------------------------------------

    # compare_log_and_linear(collapsed_df, "Area Score", plot=False)
    logistic_features2 = ['Log Area Score', 'Pressure 1.2', 
                                    'Median SYS_BP']

    # Make models
    predict_age_with_linear_regression(collapsed_df, logistic_features2)
    logistic_model2 = perform_logistic_regression(collapsed_df, logistic_features2, target)
    # logistic_model_eval2 = perform_logistic_regression_and_evaluate(collapsed_df, logistic_features2, target)

    make_roc_curve_one_var(collapsed_df, 'Log Area Score', target='Age', flip = True)
    make_roc_curve_one_var(collapsed_df, 'Area Score', target='Age', flip = True)
    # make_roc_curve_one_var(collapsed_df, 'Log Pressure 1.2', target='Age', flip = False)

    # # Calculate AUC with age threshold of 50
    # y_true = (collapsed_df['Age'] > 50).astype(int)
    # y_scores = logistic_model2.predict(collapsed_df[logistic_features2])
    # auc = calculate_auc(y_true, y_scores, features = logistic_features2, plot=True)

    # # plot confusion matrix for logistic model
    # plot_confusion_matrix(y_true, y_scores, threshold=50, features = logistic_features2, class_names=['Under 50', 'Over 50'])

    # # plot ROC curve for logistic model
    # fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    # plt.figure()
    # plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
    # plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic (ROC) for Logistic Regression')
    # plt.legend(loc="lower right")
    # plt.show()

    

    # -------------------------------------------------------------------------------------------------------

    return 0

# def calculate_cdf_area(data, start=10, end=700):
#     # Generate a linear space from the min to max velocity in the range of interest
#     x = np.linspace(start, end, num=500)
    
#     # Calculate the CDF using the empirical data
#     cdf = np.interp(x, np.sort(data), np.linspace(0, 1, len(data)))
    
#     # Calculate the area under the CDF curve using Simpson's rule
#     area = simps(cdf, x)
#     return area

# def calculate_distance_score(data):
#     # Calculate the CDF area for the entire dataset
#     entire_dataset_area = calculate_cdf_area(data)

#     # Calculate the CDF area for each participant and compute the distance score
#     participant_scores = data.groupby('Participant')['Corrected Velocity'].apply(lambda x: entire_dataset_area - calculate_cdf_area(x))

#     return participant_scores

def empirical_cdf(data):
    """Generates the empirical CDF for a dataset."""
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(data) + 1) / len(data)
    return sorted_data, cdf

def plot_ks_statistic(sample, reference):
    """Plots the empirical CDFs of a sample and a reference dataset, highlighting the KS statistic."""
    # Compute empirical CDFs
    sample_sorted, sample_cdf = empirical_cdf(sample)
    reference_sorted, reference_cdf = empirical_cdf(reference)
    
    # Calculate KS statistic and the corresponding x-value
    differences = np.abs(sample_cdf - np.interp(sample_sorted, reference_sorted, reference_cdf))
    ks_statistic = np.max(differences)
    ks_x = sample_sorted[np.argmax(differences)]
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(sample_sorted, sample_cdf, label='Sample CDF', linestyle='--', color='blue')
    plt.plot(reference_sorted, reference_cdf, label='Reference CDF', color='green')
    
    # Highlight the KS statistic
    plt.fill_betweenx([0, 1], ks_x, ks_x + ks_statistic, color='red', alpha=0.3, label=f'KS Statistic = {ks_statistic:.4f}')
    
    plt.title('Empirical CDFs and KS Statistic')
    plt.xlabel('Value')
    plt.ylabel('CDF')
    plt.legend()
    plt.grid(True)
    plt.show()

def kolmogorov_smirnov_test_per_part(entire_dataset, reference, plot = False):
    # Create the empirical CDF function from the dataset
    ecdf = empirical_cdf_fn(reference['Corrected Velocity'])

    ks_vals = pd.DataFrame(columns=['Participant', 'KS Statistic', 'KS P-Value'])
    # Perform the KS test comparing the sample to the empirical CDF
    for participant in entire_dataset['Participant'].unique():
        participant_velocities = entire_dataset[entire_dataset['Participant'] == participant]['Corrected Velocity']
        # Perform the KS test comparing the sample to the empirical CDF
        ks_statistic, p_value = kstest(participant_velocities, ecdf)
        ks_vals = ks_vals.append({'Participant': participant, 'KS Statistic': ks_statistic, 'KS P-Value': p_value}, ignore_index=True)
    if plot:
        # plot the KS statistic and p-value for each participant on two different subplots
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.bar(ks_vals['Participant'], ks_vals['KS Statistic'], width=0.5)
        plt.xlabel('Participant')
        plt.ylabel('KS Statistic')
        plt.title('KS Statistic for Each Participant')
        plt.xticks(rotation=45)
        plt.subplot(1, 2, 2)
        plt.bar(ks_vals['Participant'], ks_vals['KS P-Value'], width=0.5)
        plt.xlabel('Participant')
        plt.ylabel('P-Value')
        plt.title('P-Value for Each Participant')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        return 0

    # Assuming data_above_50 and data_below_50 are arrays containing all measurements
def bootstrap_test(data1, data2, n_iterations=1000):
    size = min(len(data1), len(data2))
    p_values = []
    
    for i in range(n_iterations):
        sample1 = np.random.choice(data1, size=size, replace=True)
        sample2 = np.random.choice(data2, size=size, replace=True)
        stat, p = mannwhitneyu(sample1, sample2)
        p_values.append(p)
    
    plt.hist(p_values, bins=30, edgecolor='k', alpha=0.7)
    plt.xlabel('p-value')
    plt.ylabel('Frequency')
    plt.title('Bootstrap p-value Distribution')
    plt.show()

    return np.mean(p_values)


def calculate_cdf_area(data, start=10, end=700):
    data = data['Corrected Velocity']
    log_data = np.log1p(data)
    data_sorted = np.sort(data)
    log_data_sorted = np.sort(log_data)
    p = 1. * np.arange(len(data)) / (len(data) - 1)
    p_log = 1. * np.arange(len(log_data)) / (len(log_data) - 1)
    
    # Interpolate to find CDF values at start and end points if necessary
    start_cdf = np.interp(start, data_sorted, p)
    start_cdf_log = np.interp(np.log1p(start), log_data_sorted, p_log)
    end_cdf = np.interp(end, data_sorted, p)
    end_cdf_log = np.interp(np.log1p(end), log_data_sorted, p_log)
    
    # Calculate the area under the CDF curve using Simpson's rule within the range
    x = np.linspace(start, end, num=500)  # More points for a smoother curve and more accurate integration
    x_log = np.linspace(np.log1p(start), np.log1p(end), num=500)
    cdf_values = np.interp(x, data_sorted, p)
    cdf_values_log = np.interp(x_log, log_data_sorted, p_log)
    area = simps(cdf_values, x)
    area_log = simps(cdf_values_log, x_log)
    
    return area, area_log

def calculate_area_score(data, start=10, end=700, plot = False, verbose = False, log = False):
    area, area_log = calculate_cdf_area(data, start, end)
    if verbose:
        print(f'Area: {area:.2f}, Log Area: {area_log:.2f}')
    area_scores = []
    for participant in data['Participant'].unique():
        participant_df = data[data['Participant'] == participant]
        participant_area, participant_area_log = calculate_cdf_area(participant_df)
        if verbose:
            print(f'Participant {participant} has a CDF area of {participant_area:.2f} and a log CDF area of {participant_area_log:.2f}')
        area_scores.append([participant, participant_area-area, participant_area_log-area_log])
    # plot area scores
    area_scores_df = pd.DataFrame(area_scores, columns=['Participant', 'Area Score', 'Log Area Score'])
    plt.figure(figsize=(10, 6))
    if log:
        area_scores_df = area_scores_df.sort_values(by='Log Area Score', ascending=False)
        plt.bar(area_scores_df['Participant'], area_scores_df['Log Area Score'], width=0.5)
        plt.ylabel('Log Area Score')
        plt.title('Log Area Score for Each Participant')
    else:
        area_scores_df = area_scores_df.sort_values(by='Area Score', ascending=False)
        plt.bar(area_scores_df['Participant'], area_scores_df['Area Score'], width=0.5)
        plt.ylabel('Area Score')
        plt.title('Area Score for Each Participant')
    plt.xlabel(f'Participant')
    plt.xticks(rotation=45)
    if plot:
        plt.show()
    else:
        plt.close()
    return area_scores_df

def plot_medians_pvals(summary_df_nhp_video_medians):
    if 'Sex' not in summary_df_nhp_video_medians.columns:
        raise ValueError("DataFrame must include 'Sex' column for ANOVA analysis.")

    print('Summary df nhp video medians')
    print(summary_df_nhp_video_medians.columns)
    # Rename 'Video Median Velocity' column to avoid spaces
    summary_df_nhp_video_medians = summary_df_nhp_video_medians.rename(columns={'Video Median Velocity': 'Video_Median_Velocity'})
    # Calculate the median velocity per participant
    medians_nhp_vidmed = summary_df_nhp_video_medians.groupby('Participant')['Video_Median_Velocity'].median().reset_index()

    # Merge this back with the original data to get age and sex information
    merged_data = pd.merge(medians_nhp_vidmed, summary_df_nhp_video_medians[['Participant', 'Age', 'Sex']], on='Participant', how='left')
    merged_data = merged_data.drop_duplicates().reset_index(drop=True)

    # Define age groups
    merged_data['Age_Group'] = merged_data['Age'].apply(lambda x: 'Above 50' if x >= 50 else 'Below 50')

    # Statistical test
    above_50 = merged_data[merged_data['Age_Group'] == 'Above 50']['Video_Median_Velocity']
    below_50 = merged_data[merged_data['Age_Group'] == 'Below 50']['Video_Median_Velocity']
    stat, p_value = mannwhitneyu(above_50, below_50)

    # Visualization
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Age_Group', y='Video_Median_Velocity', data=merged_data, palette="Set3", boxprops=dict(alpha=.3))
    sns.swarmplot(x='Age_Group', y='Video_Median_Velocity', data=merged_data, color='black')
    plt.title('Comparison of Median Blood Flow Velocities by Age Group')
    plt.xlabel('Age Group')
    plt.ylabel('Median Blood Flow Velocity')
    plt.annotate(f'p-value = {p_value:.3f}', xy=(0.5, 0.95), xycoords='axes fraction', ha='center', va='top',
                 fontsize=12, color='red')
    plt.show()

    # Print the p-value
    print(f"The p-value for the comparison between groups is: {p_value}")

    # Fit model for ANOVA including Sex
    model = ols('Video_Median_Velocity ~ C(Age_Group) + C(Sex) + C(Age_Group):C(Sex)', data=merged_data).fit()

    # ANOVA table
    anova_results = sm.stats.anova_lm(model, typ=2)
    print(anova_results)

    return 0

def plot_results_with_annotations(data):
    plt.figure(figsize=(10, 6))
    
    # Creating a box plot
    sns.boxplot(x='Age_Group', y='Video_Median_Velocity', hue='Sex', data=data, palette="Set2")

    # Adding titles and labels
    plt.title('Impact of Age Group and Sex on Video Median Velocity')
    plt.xlabel('Age Group')
    plt.ylabel('Median Video Velocity')

    # Annotating with p-values
    # Adjust the positions according to your plot structure
    plt.text(0.5, 300, f'p = 2.23e-08', horizontalalignment='center', color='black', weight='semibold')
    plt.text(0.5, 280, f'Age*BP p = 0.00087', horizontalalignment='center', color='red', weight='semibold')
    plt.text(0.5, 260, f'Sex*BP p = 0.00051', horizontalalignment='center', color='blue', weight='semibold')

    plt.legend(title='Sex')
    plt.show()


# def analyze_velocity_influence(summary_df):
#     if 'Sex' not in summary_df.columns or 'SYS_BP' not in summary_df.columns:
#         raise ValueError("DataFrame must include both 'Sex' and 'SYS_BP' columns for the analysis.")

#     # Prepare your data: Ensure there are no missing values, etc.
#     summary_df.dropna(subset=['Video Median Velocity', 'Age', 'Sex', 'SYS_BP'], inplace=True)

#     # rename 'Video Median Velocity' column to avoid spaces
#     summary_df = summary_df.rename(columns={'Video Median Velocity': 'Video_Median_Velocity'})

#     # Define age groups
#     summary_df['Age_Group'] = summary_df['Age'].apply(lambda x: 'Above 50' if x >= 50 else 'Below 50')

#     # Fit the ANOVA model with SYS_BP included
#     model = ols('Video_Median_Velocity ~ C(Age_Group) + C(Sex) + SYS_BP + C(Age_Group):C(Sex) + C(Age_Group):SYS_BP + C(Sex):SYS_BP', data=summary_df).fit()

#     # Display ANOVA table
#     anova_results = sm.stats.anova_lm(model, typ=2)
#     print(anova_results)
#     plot_results_with_annotations(summary_df)

#     return model

def perform_anova_analysis(df, variable ='Age', plot = True):
    # Calculate the median velocity per participant
    participant_medians = df.groupby('Participant').agg({
        'Video Median Velocity': 'median',  # Assume your velocity column is named 'Video Median Velocity'
        'Age': 'first',  # Assumes each participant's age is constant across rows
        'Sex': 'first',  # Assumes sex is constant
        'SYS_BP': 'median'  # Assumes systolic blood pressure is constant
    }).reset_index()

    # Rename the aggregated velocity for clarity
    participant_medians.rename(columns={'Video Median Velocity': 'Participant_Median_Velocity'}, inplace=True)

    # Fit model for ANOVA including SYS_BP
    model = ols('Participant_Median_Velocity ~ Age + C(Sex) + SYS_BP + Age:C(Sex) + Age:SYS_BP + C(Sex):SYS_BP', data=participant_medians).fit()

    # ANOVA table
    anova_results = sm.stats.anova_lm(model, typ=2)
    print("ANOVA Results:")
    print(anova_results)

    # # Visualization
    # plt.figure(figsize=(12, 8))
    # sns.boxplot(x='Age', y='Participant_Median_Velocity', hue='Sex', data=participant_medians, palette='Set2')
    # plt.title('Impact of Age and Sex on Median Participant Velocity')
    # plt.xlabel('Age')
    # plt.ylabel('Median Participant Velocity')
    # plt.legend(title='Sex')
    # plt.show()

    # Assuming 'participant_medians' is the DataFrame prepared earlier with median velocities and demographic data
    # Reclassify 'Age' into 'Above 50' and 'Below 50'
    participant_medians['Age_Group'] = participant_medians['Age'].apply(lambda x: 'Above 50' if x >= 50 else 'Below 50')

    # # Plotting
    # plt.figure(figsize=(12, 8))
    # sns.boxplot(x='Age_Group', y='Participant_Median_Velocity', hue='Sex', data=participant_medians, palette='Set2')
    # plt.title('Impact of Age Group and Sex on Median Participant Velocity')
    # plt.xlabel('Age Group')
    # plt.ylabel('Median Participant Velocity')

    # # Adding annotations for significant results, adjust these based on your specific p-values
    # plt.text(0.5, participant_medians['Participant_Median_Velocity'].max() * 0.9, f'Age Group p = 0.000083', horizontalalignment='center', color='black', weight='semibold')
    # plt.text(0.5, participant_medians['Participant_Median_Velocity'].max() * 0.85, f'Age*Sex p = 0.001886', horizontalalignment='center', color='red', weight='semibold')
    # plt.text(0.5, participant_medians['Participant_Median_Velocity'].max() * 0.8, f'Age*BP p = 0.002298', horizontalalignment='center', color='blue', weight='semibold')

    # plt.legend(title='Sex')
    # plt.show()

    # Assuming 'participant_medians' is your DataFrame with median velocities and other demographic data
    # Adjust the age column to create two groups: Above 50 and Below 50
    participant_medians['Age_Group'] = participant_medians['Age'].apply(lambda x: 'Above 50' if x >= 50 else 'Below 50')
    participant_medians['SYS_BP_Group'] = np.where(participant_medians['SYS_BP'] < 120, '<120', '≥120')
    participant_medians['Sex_Group'] = np.where(participant_medians['Sex'] == 'M', 'M', 'F')

    # print the two group medians
    # Calculate the two group medians
    above_50_median = participant_medians.loc[participant_medians['Age_Group'] == 'Above 50', 'Participant_Median_Velocity'].median()
    below_50_median = participant_medians.loc[participant_medians['Age_Group'] == 'Below 50', 'Participant_Median_Velocity'].median()
    above_120_median = participant_medians.loc[participant_medians['SYS_BP_Group'] == '≥120', 'Participant_Median_Velocity'].median()
    below_120_median = participant_medians.loc[participant_medians['SYS_BP_Group'] == '<120', 'Participant_Median_Velocity'].median()
    male_median = participant_medians.loc[participant_medians['Sex_Group'] == 'M', 'Participant_Median_Velocity'].median()
    female_median = participant_medians.loc[participant_medians['Sex_Group'] == 'F', 'Participant_Median_Velocity'].median()

    # Calculate standard deviations for different groups
    above_50_std = participant_medians.loc[participant_medians['Age_Group'] == 'Above 50', 'Participant_Median_Velocity'].std()
    below_50_std = participant_medians.loc[participant_medians['Age_Group'] == 'Below 50', 'Participant_Median_Velocity'].std()
    above_120_std = participant_medians.loc[participant_medians['SYS_BP_Group'] == '≥120', 'Participant_Median_Velocity'].std()
    below_120_std = participant_medians.loc[participant_medians['SYS_BP_Group'] == '<120', 'Participant_Median_Velocity'].std()
    male_std = participant_medians.loc[participant_medians['Sex_Group'] == 'M', 'Participant_Median_Velocity'].std()
    female_std = participant_medians.loc[participant_medians['Sex_Group'] == 'F', 'Participant_Median_Velocity'].std()

    # Print the two group medians and standard deviations
    print(f"Median Velocity for Age Group Above 50: {above_50_median:.2f} (Std: {above_50_std:.2f})")
    print(f"Median Velocity for Age Group Below 50: {below_50_median:.2f} (Std: {below_50_std:.2f})")
    print(f"Difference in Medians: {above_50_median - below_50_median:.2f}")
    print(f"Percentage Increase: {((above_50_median - below_50_median) / below_50_median) * 100:.2f}%")

    print(f"Median Velocity for SYS_BP Group ≥120: {above_120_median:.2f} (Std: {above_120_std:.2f})")
    print(f"Median Velocity for SYS_BP Group <120: {below_120_median:.2f} (Std: {below_120_std:.2f})")

    print(f"Median Velocity for Males: {male_median:.2f} (Std: {male_std:.2f})")
    print(f"Median Velocity for Females: {female_median:.2f} (Std: {female_std:.2f})")



    if plot:
        # Plotting
        plt.figure(figsize=(10, 6))
        if variable == 'Age':
            boxplot = sns.boxplot(x='Age_Group', y='Participant_Median_Velocity', data=participant_medians, palette='Set3')
            # sns.stripplot(x='Age_Group', y='Participant_Median_Velocity', data=participant_medians, color='black', jitter=0.1, size=5, alpha=0.6)
            sns.swarmplot(x='Age_Group', y='Participant_Median_Velocity', data=participant_medians, color='black', size=5, alpha=0.7)

            plt.title('Impact of Age Group on Median Participant Velocity')
            plt.xlabel('Age Group')
        elif variable == 'SYS_BP':
            boxplot = sns.boxplot(x='SYS_BP_Group', y='Participant_Median_Velocity', data=participant_medians, palette='Set1')
            # sns.stripplot(x='SYS_BP', y='Participant_Median_Velocity', data=participant_medians, color='black', jitter=0.1, size=5, alpha=0.6)
            sns.swarmplot(x='SYS_BP_Group', y='Participant_Median_Velocity', data=participant_medians, color='black', size=5, alpha=0.7)

            plt.title('Impact of Systolic Blood Pressure on Median Participant Velocity')
            plt.xlabel('Systolic Blood Pressure')
        elif variable == 'Sex':
            boxplot = sns.boxplot(x='Sex_Group', y='Participant_Median_Velocity', data=participant_medians, palette='Set2')
            # sns.stripplot(x='SYS_BP', y='Participant_Median_Velocity', data=participant_medians, color='black', jitter=0.1, size=5, alpha=0.6)
            sns.swarmplot(x='Sex_Group', y='Participant_Median_Velocity', data=participant_medians, color='black', size=5, alpha=0.7)

            plt.title('Impact of Biological Sex on Median Participant Velocity')
            plt.xlabel('Sex')
        plt.ylabel('Median Participant Velocity')

        # Annotate with p-value, assuming you already have it calculated or from previous analysis (p = 0.000083 in your case)
        # Adding a star (*) to indicate statistical significance
        if variable == 'Age':
            p_value = 0.000083
            significance = "*" if p_value < 0.05 else "ns"  # ns stands for not significant
            plt.text(0.5, participant_medians['Participant_Median_Velocity'].max() * 0.95, f'p = {p_value:.5f} {significance}', horizontalalignment='center', color='black', weight='semibold')

        plt.show()

    return model
    
def summarize_set01(filepath='C:\\Users\\gt8mar\\capillary-flow\\metadata\\merged\\merged_metadata.csv'):
    # Load the Excel file
    df = pd.read_csv(filepath)

    # Split the 'BP' column into 'SYS_BP' and 'DIA_BP'
    df[['SYS_BP', 'DIA_BP']] = df['BP'].str.split('/', expand=True).astype(float)

    # Filter rows for 'set01'
    set01_df = df[df['SET'] == 'set01']

    # Calculate necessary statistics
    results = {
        'Group': ['All', 'Below fifty', 'Fifty and above'],
        'Average Age': [],
        'Std Age': [],
        'Average SYS_BP': [],
        'Std SYS_BP': [],
        'Average DIA_BP': [],
        'Std DIA_BP': [],
        'Average Pulse': [],
        'Std Pulse': [],
        'Male Count': [],
        'Female Count': []
    }

    # Define age groups
    all_ages = set01_df
    below_fifty = set01_df[set01_df['Age'] < 50]
    fifty_and_above = set01_df[set01_df['Age'] >= 50]

    for group in [all_ages, below_fifty, fifty_and_above]:
        results['Average Age'].append(group['Age'].mean())
        results['Std Age'].append(group['Age'].std())
        results['Average SYS_BP'].append(group['SYS_BP'].mean())
        results['Std SYS_BP'].append(group['SYS_BP'].std())
        results['Average DIA_BP'].append(group['DIA_BP'].mean())
        results['Std DIA_BP'].append(group['DIA_BP'].std())
        results['Average Pulse'].append(group['Pulse'].mean())
        results['Std Pulse'].append(group['Pulse'].std())
        results['Male Count'].append(group[group['Sex'] == 'M'].shape[0])
        results['Female Count'].append(group[group['Sex'] == 'F'].shape[0])

    # Create a DataFrame from the results dictionary
    table_fig = pd.DataFrame(results)
    return table_fig

def extract_capillary(image_path):
    image_path = image_path.replace('.tiff', '').replace('.png', '')
    image_path_list  = image_path.split('_')
    capillary_name = image_path_list[-1]
    return capillary_name



def main(verbose = False):
    if platform.system() == 'Windows':
        if 'gt8mar' in os.getcwd():
            path = 'C:\\Users\\gt8mar\\capillary-flow\\results\\summary_df_test.csv'
        else:
            path = 'C:\\Users\\gt8ma\\capillary-flow\\results\\summary_df_test.csv'
    else:
        path = '/hpc/projects/capillary-flow/results/summary_df_test.csv'

    summary_df = pd.read_csv(path)
    classified_kymos_df = pd.read_csv('C:\\Users\\gt8ma\\capillary-flow\\classified_kymos.csv')
    classified_kymos_df['Capillary'] = classified_kymos_df['Image_Path'].apply(extract_capillary)
    
    # Merge the dataframes on the common columns
    summary_df = pd.merge(summary_df, classified_kymos_df[['Participant', 'Date', 'Location', 'Video', 'Capillary', 'Classified_Velocity']], 
                         on=['Participant', 'Date', 'Location', 'Video', 'Capillary'], how='outer')
    
    # Save or display the resulting dataframe
    # merged_df.to_csv('C:\\Users\\gt8ma\\capillary-flow\\merged_csv.csv', index=False)

    # If there is a value in "Classified Velocity" overwrite the value in "Corrected Velocity" with that value:
    summary_df['Corrected Velocity'] = np.where(summary_df['Classified_Velocity'].notnull(), summary_df['Classified_Velocity'], summary_df['Corrected Velocity'])
    summary_df.to_csv('C:\\Users\\gt8ma\\capillary-flow\\merged_csv2.csv', index=False)
    summary_df = summary_df.drop(columns=['Capillary'])
    summary_df = summary_df.rename(columns={'Capillary_new': 'Capillary'})

    
    old_subset = summary_df[summary_df['Age'] > 50]
    # drop nan values
    old_subset_no_nan = old_subset.dropna(subset=['Corrected Velocity'])
    young_subset = summary_df[summary_df['Age'] <= 50]
    # drop nan values
    young_subset_no_nan = young_subset.dropna(subset=['Corrected Velocity'])

    # plot_histograms(summary_df, 'Age')
    # plot_histograms(summary_df, 'SYS_BP')
    stat, p = mannwhitneyu(old_subset_no_nan['Corrected Velocity'], young_subset_no_nan['Corrected Velocity'], alternative='two-sided')      # could also use 'less' or 'greater'
    print('Statistics=%.3f, p=%.5f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Same distribution (fail to reject H0)')
    else:
        print('Different distribution (reject H0)')
    
    # # print(summary_df.head())

    # plot_densities(summary_df)
    # plot_densities_pressure(summary_df)
    # plot_hist_pressure(summary_df, density=True)
    
    # create a subset of summary_df with no pressure values greater than 1.2
    summary_df_no_high_pressure = summary_df[summary_df['Pressure'] <= 1.2]
    old_nhp = summary_df_no_high_pressure[summary_df_no_high_pressure['Age'] > 50]
    young_nhp = summary_df_no_high_pressure[summary_df_no_high_pressure['Age'] <= 50]
    normbp_nhp = summary_df_no_high_pressure[summary_df_no_high_pressure['SYS_BP'] <= 120]
    highbp_nhp = summary_df_no_high_pressure[summary_df_no_high_pressure['SYS_BP'] > 120]

    # compute difference in median for old and young
    old_median = old_nhp['Corrected Velocity'].median()
    young_median = young_nhp['Corrected Velocity'].median()
    print(f'Old Median: {old_median}, Young Median: {young_median}')

    # # compute p value for difference in median for old and young
    # stat, p = mannwhitneyu(old_nhp['Corrected Velocity'], young_nhp['Corrected Velocity'])      # could also use 'less' or 'greater'  , alternative='two-sided'
    # print('Statistics=%.3f, p=%.5f' % (stat, p))

    # plot_box_and_whisker(summary_df_no_high_pressure, old_nhp, young_nhp, column = 'Corrected Velocity', variable = 'Age', log_scale=True)
    # plot_box_and_whisker(summary_df_no_high_pressure, highbp_nhp, normbp_nhp, column = 'Corrected Velocity', variable='SYS_BP', log_scale=True)
    # plot_violin(summary_df_no_high_pressure, old_nhp, young_nhp, 'Corrected Velocity', True)
    # plot_hist_pressure(summary_df_no_high_pressure, density=True)
    # plot_densities(summary_df_no_high_pressure)

    # plot_hist_specific_pressure(summary_df, 0.2, density=True, hist=False)
    # plot_hist_specific_pressure(summary_df, 0.8, density=True, hist=False)
    # plot_hist_specific_pressure(summary_df, 1.2, density=True, hist=False) 

    # plot_hist_comp_pressure(summary_df, density=True, hist=False)
    # plot_densities(summary_df_no_high_pressure)
    # plot_cdf(summary_df_no_high_pressure['Corrected Velocity'], subsets= [summary_df_no_high_pressure[summary_df_no_high_pressure['Age'] > 50]['Corrected Velocity'], summary_df_no_high_pressure[summary_df_no_high_pressure['Age'] <= 50]['Corrected Velocity']], labels=['Entire Dataset', 'Old', 'Young'], title = 'CDF Comparison of velocities by Age')    
    # plot_cdf(summary_df_no_high_pressure['Corrected Velocity'], subsets= [summary_df_no_high_pressure[summary_df_no_high_pressure['SYS_BP'] > 120]['Corrected Velocity'], summary_df_no_high_pressure[summary_df_no_high_pressure['SYS_BP'] <= 120]['Corrected Velocity']], labels=['Entire Dataset', 'High BP', 'Normal BP'], title = 'CDF Comparison of velocities by BP nhp')
    
    # plot cdf for high bp old, high bp young, low bp old, low bp young
    highBP_old = summary_df[(summary_df['SYS_BP'] > 120) & (summary_df['Age'] > 50)]['Corrected Velocity']
    highBP_young = summary_df[(summary_df['SYS_BP'] > 120) & (summary_df['Age'] <= 50)]['Corrected Velocity']
    normBP_old = summary_df[(summary_df['SYS_BP'] <= 120) & (summary_df['Age'] > 50)]['Corrected Velocity']
    normBP_young = summary_df[(summary_df['SYS_BP'] <= 120) & (summary_df['Age'] <= 50)]['Corrected Velocity']

    highBP_old_nhp = summary_df_no_high_pressure[(summary_df_no_high_pressure['SYS_BP'] > 120) & (summary_df_no_high_pressure['Age'] > 50)]['Corrected Velocity']
    highBP_young_nhp = summary_df_no_high_pressure[(summary_df_no_high_pressure['SYS_BP'] > 120) & (summary_df_no_high_pressure['Age'] <= 50)]['Corrected Velocity']
    normBP_old_nhp = summary_df_no_high_pressure[(summary_df_no_high_pressure['SYS_BP'] <= 120) & (summary_df_no_high_pressure['Age'] > 50)]['Corrected Velocity']
    normBP_young_nhp = summary_df_no_high_pressure[(summary_df_no_high_pressure['SYS_BP'] <= 120) & (summary_df_no_high_pressure['Age'] <= 50)]['Corrected Velocity']

    # plot_cdf(summary_df_no_high_pressure['Corrected Velocity'], subsets= [highBP_old_nhp, highBP_young_nhp, normBP_old_nhp, normBP_young_nhp], labels=['Entire Dataset', 'High BP Old', 'High BP Young', 'Normal BP Old', 'Normal BP Young'], title = 'CDF Comparison of velocities by Age and BP')
    # plot_cdf_comp_pressure(summary_df)

    
    area_scores_df = calculate_area_score(summary_df_no_high_pressure, log = True, plot=False)
    # add area scores to summary_df_no_high_pressure
    summary_df_no_high_pressure = summary_df_no_high_pressure.merge(area_scores_df, on='Participant', how='inner')

   
    

    # # plot area score vs age scatter
    # plt.figure(figsize=(10, 6))
    # plt.scatter(summary_df_no_high_pressure['Age'], summary_df_no_high_pressure['Log Area Score'])
    # plt.xlabel('Age')
    # plt.ylabel('Log Area Score')
    # plt.title('Log Area Score vs. Age')
    # plt.show()






    # summary_metrics = calculate_metrics(summary_df['Corrected Velocity'])
    # print(summary_metrics)
    
    skewness = []
    kurtosis = []
    ecdf_fn = empirical_cdf_fn(summary_df['Corrected Velocity'])
    ks_statistic_df = pd.DataFrame(columns=['Participant', 'KS Statistic', 'KS P-Value', 'EMD Score'])
    for participant in summary_df['Participant'].unique():
        participant_df = summary_df[summary_df['Participant'] == participant]
        participant_df_nhp = summary_df_no_high_pressure[summary_df_no_high_pressure['Participant'] == participant]
        participant_metrics = calculate_metrics(participant_df['Corrected Velocity'])
        skewness.append([participant,participant_metrics['skewness']])
        kurtosis.append([participant,participant_metrics['kurtosis']])
        ks_statistic, p_value = kstest(participant_df['Corrected Velocity'], ecdf_fn)
        emd_score = wasserstein_distance(participant_df['Corrected Velocity'], summary_df['Corrected Velocity'])
        ks_statistic_df = pd.concat([ks_statistic_df, pd.DataFrame({'Participant': [participant], 'KS Statistic': [ks_statistic], 'KS P-Value': [p_value], 'EMD Score': [emd_score]})])
        # plot_ks_statistic(participant_df['Corrected Velocity'], summary_df['Corrected Velocity'])
    
        # # Plot density
        # sns.kdeplot(summary_df['Corrected Velocity'], label='Entire Dataset', fill=True)
        # sns.kdeplot(participant_df['Corrected Velocity'], label=participant, fill=True, alpha=0.5)
        # plt.legend()
        # plt.title('Density Plot of Entire Dataset vs. Subset')
        # plt.show()

        # # Plot CDF
        # plot_cdf(summary_df['Corrected Velocity'], subsets= [participant_df['Corrected Velocity']], labels=['Entire Dataset', participant], title = f'CDF Comparison of velocities for {participant}')
        # plot_cdf(summary_df_no_high_pressure['Corrected Velocity'], subsets= [participant_df_nhp['Corrected Velocity']], labels=['Entire Dataset', participant], title = f'CDF Comparison of velocities for {participant} nhp', write=False)
        # plot_cdf_comp_pressure(participant_df)
        # plot_cdf_comp_pressure(participant_df_nhp)    
    
    # merge ks statistic df with summary df
    summary_df_no_high_pressure = summary_df_no_high_pressure.merge(ks_statistic_df, on='Participant', how='inner')

    # Input sex values for each participant 
    sex_list = [['part09', 'F'], ['part10', 'F'], ['part11', 'M'], ['part12', 'F'], ['part13', 'M'], ['part14', 'M'], 
                ['part15', 'M'], ['part16', 'F'], ['part17', 'M'], ['part18', 'F'], ['part19', 'M'], ['part20', 'F'], 
                ['part21', 'M'], ['part22', 'M'], ['part23', 'F'], ['part25', 'F'], ['part26', 'M'], ['part27', 'F']]
    sex_df = pd.DataFrame(sex_list, columns=['Participant', 'Sex'])
    summary_df_no_high_pressure = summary_df_no_high_pressure.merge(sex_df, on='Participant', how='inner')

    male_subset = summary_df_no_high_pressure[summary_df_no_high_pressure['Sex']=='M']
    female_subset = summary_df_no_high_pressure[summary_df_no_high_pressure['Sex']=='F']

    summary_df_nhp_video_medians = calculate_video_median_velocity(summary_df_no_high_pressure)
    old_nhp_video_medians = summary_df_nhp_video_medians[summary_df_nhp_video_medians['Age'] > 50]
    young_nhp_video_medians = summary_df_nhp_video_medians[summary_df_nhp_video_medians['Age'] <= 50]
    normbp_nhp_video_medians = summary_df_nhp_video_medians[summary_df_nhp_video_medians['SYS_BP'] <= 120]
    highbp_nhp_video_medians = summary_df_nhp_video_medians[summary_df_nhp_video_medians['SYS_BP'] > 120]

    """
    -------- This is where I am running stuff rn --------
    """
    # remove part21, part22, part24
    summary_df_nhp_video_medians = summary_df_nhp_video_medians[~summary_df_nhp_video_medians['Participant'].isin(['part21', 'part22', 'part24'])]

    # plot_medians_pvals(summary_df_nhp_video_medians)
    # analyze_velocity_influence(summary_df_nhp_video_medians)
    perform_anova_analysis(summary_df_nhp_video_medians, plot = False)
    # table_fig = summarize_set01()
    # print(table_fig)
    # plot_box_and_whisker(summary_df_nhp_video_medians, highbp_nhp_video_medians, normbp_nhp_video_medians, column = 'Video Median Velocity', variable='SYS_BP', log_scale=True)
    plot_cdf(summary_df_nhp_video_medians['Video Median Velocity'], subsets= [old_nhp_video_medians['Video Median Velocity'], young_nhp_video_medians['Video Median Velocity']], labels=['Entire Dataset', 'Old', 'Young'], title = 'CDF Comparison of Video Median Velocities by Age',
             write =True, variable='Age')
    # plot_individual_cdfs(summary_df_nhp_video_medians)
    plot_cdf(summary_df_nhp_video_medians['Video Median Velocity'], subsets= [highbp_nhp_video_medians['Video Median Velocity'], normbp_nhp_video_medians['Video Median Velocity']], labels=['Entire Dataset', 'High BP', 'Normal BP'], title = 'CDF Comparison of Video Median Velocities by BP nhp',
             write = True, variable='SYS_BP')
    plot_cdf(summary_df_no_high_pressure['Corrected Velocity'], 
             subsets=[male_subset['Corrected Velocity'], female_subset['Corrected Velocity']],
                labels=['Entire Dataset', 'Male', 'Female'], title='CDF Comparison of velocities by Sex', 
                normalize = False, variable='Sex', write=True)

    # plot_cdf_comp_pressure(summary_df_nhp_video_medians)

    # make subset of age for participants under 25
    # under_25 = summary_df_no_high_pressure[summary_df_no_high_pressure['Age'] < 25]
    # plot_velocity_vs_diameter(under_25, hue = 'Pressure')
    # plot_velocity_vs_diameter2(summary_df_no_high_pressure, hue = 'Age')
   
    """ Below here is dimless stuff"""
    # summary_df_dimless = add_dimensionless_velocity(summary_df_no_high_pressure)
    # dimensionless_velocities = summary_df_dimless['Dimensionless Velocity']
    # # print number of nan values
    # print(f'Number of nan values in dimensionless velocities: {dimensionless_velocities.isna().sum()}')
    # # plot average viscocities for each participant
    # # replace 'Corrected Velocity' values with 'Dimensionless Velocity' in summary_df_dimless
    # # summary_df_dimless['Corrected Velocity'] = np.log(summary_df_dimless['Corrected Velocity']) /summary_df_dimless['Dimensionless Velocity']
    # plot_cdf(summary_df_dimless['Dimensionless Velocity'], subsets= [summary_df_dimless[summary_df_dimless['Age'] > 50]['Dimensionless Velocity'], summary_df_dimless[summary_df_dimless['Age'] <= 50]['Dimensionless Velocity']], labels=['Entire Dataset', 'Old', 'Young'], title = 'CDF Comparison of Dimensionless Velocities by Age', write=True, variable='Age')
    # plot_cdf(summary_df_dimless['Dimensionless Velocity'], subsets= [summary_df_dimless[summary_df_dimless['SYS_BP'] > 120]['Dimensionless Velocity'], summary_df_dimless[summary_df_dimless['SYS_BP'] <= 120]['Dimensionless Velocity']], labels=['Entire Dataset', 'High BP', 'Normal BP'], title = 'CDF Comparison of Dimensionless Velocities by BP', write=True, variable='SYS_BP')
    # plot_cdf(summary_df_dimless['Dimensionless Velocity'], subsets= [summary_df_dimless[summary_df_dimless['Sex'] == 'M']['Dimensionless Velocity'], summary_df_dimless[summary_df_dimless['Sex'] == 'F']['Dimensionless Velocity']], labels=['Entire Dataset', 'Male', 'Female'], title = 'CDF Comparison of Dimensionless Velocities by Sex', write=True, variable = 'Sex')
    # plot_CI(summary_df_dimless, variable = 'Age', ci_percentile=95, dimensionless = True)
    # plot_CI(summary_df_dimless, variable = 'SYS_BP', ci_percentile=95, dimensionless=True)
    # plot_CI(summary_df_dimless, variable = 'Sex', ci_percentile=95, dimensionless=True)
    
    # # make the same plots for medians:
    # summary_df_dimless_video_medians = calculate_video_median_velocity(summary_df_dimless, dimensionless=True)
    # old_nhp_video_medians_dimless = summary_df_dimless_video_medians[summary_df_dimless_video_medians['Age'] > 50]
    # young_nhp_video_medians_dimless = summary_df_dimless_video_medians[summary_df_dimless_video_medians['Age'] <= 50]
    # normbp_nhp_video_medians_dimless = summary_df_dimless_video_medians[summary_df_dimless_video_medians['SYS_BP'] <= 120]
    # highbp_nhp_video_medians_dimless = summary_df_dimless_video_medians[summary_df_dimless_video_medians['SYS_BP'] > 120]
    # plot_cdf(summary_df_dimless_video_medians['Video Median Dimensionless Velocity'], subsets= [old_nhp_video_medians_dimless['Video Median Dimensionless Velocity'], 
    #             young_nhp_video_medians_dimless['Video Median Dimensionless Velocity']], labels=['Entire Dataset', 'Old', 'Young'], 
    #             title = 'CDF Comparison of Video Median Dimensionless Velocities by Age', write =True, variable='Age')
    # plot_cdf(summary_df_dimless_video_medians['Video Median Dimensionless Velocity'], subsets= [highbp_nhp_video_medians_dimless['Video Median Dimensionless Velocity'], normbp_nhp_video_medians_dimless['Video Median Dimensionless Velocity']], labels=['Entire Dataset', 'High BP', 'Normal BP'], title = 'CDF Comparison of Video Median Dimensionless Velocities by BP nhp',
    #             write = True, variable='SYS_BP')
    # plot_cdf(summary_df_dimless_video_medians['Video Median Dimensionless Velocity'], subsets= [summary_df_dimless_video_medians[summary_df_dimless_video_medians['Sex'] == 'M']['Video Median Dimensionless Velocity'], summary_df_dimless_video_medians[summary_df_dimless_video_medians['Sex']=='F']['Video Median Dimensionless Velocity']], 
    #          labels=['Entire Dataset', 'Men', 'Women'], title='CDF Comparison of Video Median Dimensionless Velocities by Sex', write=True , variable='Sex')
    # plot_CI(summary_df_dimless_video_medians, variable = 'Age', ci_percentile=95, dimensionless = True, video_median = True)
    # plot_CI(summary_df_dimless_video_medians, variable = 'SYS_BP', ci_percentile=95, dimensionless=True, video_median = True)
    # plot_CI(summary_df_dimless_video_medians, variable = 'Sex', ci_percentile=95, dimensionless=True, video_median = True)


    """
    ------------------ back to mess ------------------
    """
    # incorrect_vels = get_bad_apples(summary_df_no_high_pressure)
    # save to csv
    # incorrect_vels.to_csv('C:\\Users\\gt8mar\\capillary-flow\\incorrect_vels.csv', index=False)

    # plot_median_velocity_of_videos(summary_df_nhp_video_medians)
    # make a copy of summary_df_nhp_video_medians where we replace 'Velocity' with 'Video Median Velocity'
    summary_df_nhp_video_medians_copy = summary_df_nhp_video_medians.copy()
    # remove the 'Corrected Velocity' column
    summary_df_nhp_video_medians_copy = summary_df_nhp_video_medians_copy.drop(columns=['Corrected Velocity'])
    # print(f'the length of summary_df_nhp_video_medians_copy is {len(summary_df_nhp_video_medians_copy)}')
    # print(f'the length of summary_df_no_high_pressure is {len(summary_df_no_high_pressure)}')
    summary_df_nhp_video_medians_copy = summary_df_nhp_video_medians_copy.rename(columns={'Video Median Velocity': 'Corrected Velocity'})
    # plot_box_whisker_pressure(summary_df_nhp_video_medians_copy, variable='Age', log_scale=False)
    # plot_CI_overlaps(summary_df_nhp_video_medians_copy, ci_percentile=95, variable='SYS_BP')
    # plot_CI_overlaps(summary_df_nhp_video_medians_copy, ci_percentile=95, variable='Age')
    # plot_CI_overlaps(summary_df_nhp_video_medians_copy, ci_percentile=95, variable='Sex')
    plot_CI(summary_df_nhp_video_medians_copy, variable = 'Sex', ci_percentile=95, write = True)
    plot_CI(summary_df_nhp_video_medians_copy, variable = 'Age', ci_percentile=95, write = True)
    plot_CI(summary_df_nhp_video_medians_copy, variable = 'SYS_BP', ci_percentile=95, write = True)

    medians_area_scores_df = calculate_area_score(summary_df_nhp_video_medians_copy, log = True, plot=False)
    # add area scores to summary_df_nhp_video_medians_copy
    summary_df_nhp_video_medians_copy = summary_df_nhp_video_medians_copy.merge(medians_area_scores_df, on='Participant', how='inner')
    # print columns of summary_df_nhp_video_medians_copy
    # print(summary_df_nhp_video_medians_copy.columns)

    

                                  
    # Plot median velocity by participant
    median_velocity_per_participant = summary_df_no_high_pressure.groupby('Participant')['Corrected Velocity'].median().sort_values()
    sorted_participant_indices = {participant: index for index, participant in enumerate(median_velocity_per_participant.index)}
    

    # plt.figure(figsize=(10, 6))
    # plt.bar(sorted_participant_indices.values(), median_velocity_per_participant.values, width=0.5)
    # plt.xlabel('Participant')
    # plt.ylabel('Median Corrected Velocity')
    # plt.title('Median Corrected Velocity for Each Participant')
    # plt.xticks(list(sorted_participant_indices.values()), list(sorted_participant_indices.keys()), rotation=45)
    # plt.show()
        
    # run_regression(summary_df_no_high_pressure)

    summary_df_nhp_video_medians_copy = summary_df_nhp_video_medians_copy.rename(columns={'Area Score_y': 'Area Score', 'Log Area Score_y': 'Log Area Score'})
    # run_regression(summary_df_nhp_video_medians_copy)
    
    # plot_CI(summary_df_no_high_pressure)        
        

    
  
    # # ####### Favorite Capillaries ######
    # if platform.system() == 'Windows':
    #     if 'gt8mar' in os.getcwd():
    #         favorite_capillaries = pd.read_excel('C:\\Users\\gt8mar\\capillary-flow\\results\\velocities\\chosen_caps.xlsx', sheet_name='Sheet1')
    #     else:
    #         favorite_capillaries = pd.read_excel('C:\\Users\\gt8ma\\capillary-flow\\results\\velocities\\chosen_caps.xlsx', sheet_name='Sheet1')
    
    # favorite_capillaries = favorite_capillaries.rename(columns={'Chosen Capillary': 'Capillary'})

    # # slice summary_df into favorite capillaries if capillary, location, and participant match
    # favorite_df = summary_df.merge(favorite_capillaries, on=['Participant', 'Location', 'Capillary'], how='inner')

    # # save to csv
    # # favorite_df.to_csv('C:\\Users\\gt8ma\\capillary-flow\\favorite_caps.csv', index=False)
    # # print(favorite_df.columns)

    # # remove part22 and part23
    # # favorite_df = favorite_df[~favorite_df['Participant'].isin(['part22', 'part23'])]
    
    # # plot_histograms(favorite_df, 'Age')
    # # plot_histograms(favorite_df, 'SYS_BP')
    
    # # plot_loc_histograms(favorite_df, 'Age')
    # # plot_loc_histograms(favorite_df, 'SYS_BP')
    # # plot_densities(favorite_df)

    # favorite_df_no_high_pressure = favorite_df[favorite_df['Pressure'] <= 1.2]
    # print(f'The length of favorite_df_no_high_pressure is {len(favorite_df_no_high_pressure)}')
    # # plot_CI(favorite_df_no_high_pressure, variable = 'Age', ci_percentile=95)
    # # plot_CI(favorite_df_no_high_pressure, variable = 'Age', method = 'mean', ci_percentile=95)
    # # plot_CI(favorite_df_no_high_pressure, variable = 'SYS_BP', ci_percentile=95)
    # # plot_CI(favorite_df_no_high_pressure, variable = 'SYS_BP', method = 'mean', ci_percentile=95)

    # # plot_box_whisker_pressure(favorite_df_no_high_pressure, 'Age', log_scale=True)
    # # plot_box_whisker_pressure(favorite_df_no_high_pressure, 'SYS_BP', log_scale=True)
    # # plot_box_whisker_pressure(favorite_df_no_high_pressure, 'Age', log_scale=False)
    # # plot_box_whisker_pressure(favorite_df_no_high_pressure, 'SYS_BP', log_scale=False)

    # # plot_hist_pressure(favorite_df_no_high_pressure, density=True)
    # # plot_densities(favorite_df_no_high_pressure)
    # # plot_cdf(favorite_df_no_high_pressure['Corrected Velocity'], subsets= [favorite_df_no_high_pressure[favorite_df_no_high_pressure['Age'] > 50]['Corrected Velocity'], favorite_df_no_high_pressure[favorite_df_no_high_pressure['Age'] <= 50]['Corrected Velocity']], labels=['Entire Dataset', 'Old', 'Young'], title = 'CDF Comparison by Age')
    # # plot_cdf(favorite_df_no_high_pressure['Corrected Velocity'], subsets= [favorite_df_no_high_pressure[favorite_df_no_high_pressure['SYS_BP'] > 120]['Corrected Velocity'], favorite_df_no_high_pressure[favorite_df_no_high_pressure['SYS_BP'] <= 120]['Corrected Velocity']], labels=['Entire Dataset', 'High BP', 'Normal BP'], title = 'CDF Comparison by BP')


    # old_fav_nhp = favorite_df_no_high_pressure[(favorite_df_no_high_pressure['Age'] > 50)]['Corrected Velocity']
    # young_fav_nhp = favorite_df_no_high_pressure[(favorite_df_no_high_pressure['Age'] <= 50)]['Corrected Velocity']

    # area_scores_fav_df = calculate_area_score(favorite_df_no_high_pressure, plot = False, log = True)
    # favorite_df_no_high_pressure = favorite_df_no_high_pressure.merge(area_scores_fav_df, on='Participant', how='inner')

    # # # plot area score vs age scatter
    # # plt.figure(figsize=(10, 6))
    # # plt.scatter(favorite_df_no_high_pressure['Age'], favorite_df_no_high_pressure['Area Score'])
    # # plt.xlabel('Age')
    # # plt.ylabel('Area Score')
    # # plt.title('Area Score vs. Age')
    # # plt.show()

    # # # plot area score vs age scatter for medians
    # # plt.figure(figsize=(10, 6))
    # # plt.scatter(summary_df_nhp_video_medians_copy['Age'], summary_df_nhp_video_medians_copy['Area Score_x'])
    # # plt.scatter(summary_df_nhp_video_medians_copy['Age'], summary_df_nhp_video_medians_copy['Area Score_y'])
    # # plt.scatter(favorite_df_no_high_pressure['Age'], favorite_df_no_high_pressure['Area Score'])
    # # plt.xlabel('Age')
    # # plt.ylabel('Area Score')
    # # plt.title('Area Score vs. Age')
    # # plt.legend(['Corrected Velocity', 'Video Median Velocity', 'Favorite Capillaries'])
    # # plt.show()


    # highBP_old_fav_nhp = favorite_df_no_high_pressure[(favorite_df_no_high_pressure['SYS_BP'] > 120) & (favorite_df_no_high_pressure['Age'] > 50)]['Corrected Velocity']
    # highBP_young_fav_nhp = favorite_df_no_high_pressure[(favorite_df_no_high_pressure['SYS_BP'] > 120) & (favorite_df_no_high_pressure['Age'] <= 50)]['Corrected Velocity']
    # normBP_old_fav_nhp = favorite_df_no_high_pressure[(favorite_df_no_high_pressure['SYS_BP'] <= 120) & (favorite_df_no_high_pressure['Age'] > 50)]['Corrected Velocity']
    # normBP_young_fav_nhp = favorite_df_no_high_pressure[(favorite_df_no_high_pressure['SYS_BP'] <= 120) & (favorite_df_no_high_pressure['Age'] <= 50)]['Corrected Velocity']

    # # plot_cdf(favorite_df_no_high_pressure['Corrected Velocity'], subsets= [highBP_old_fav_nhp, highBP_young_fav_nhp, normBP_old_fav_nhp, normBP_young_fav_nhp], labels=['Entire Dataset', 'High BP Old', 'High BP Young', 'Normal BP Old', 'Normal BP Young'], title = 'CDF Comparison by Age and BP nhp')
    # # plot_hist_specific_pressure(favorite_df_no_high_pressure, 0.2, density=True, hist=False)
    # # plot_hist_specific_pressure(favorite_df_no_high_pressure, 0.8, density=True, hist=False)
    # # plot_hist_specific_pressure(favorite_df_no_high_pressure, 1.2, density=True, hist=False)
    # # plot_cdf_comp_pressure(favorite_df_no_high_pressure)

    # # plot_hist_comp_pressure(summary_df_no_high_pressure, density=True, hist=False)

    # favorite_metrics = calculate_metrics(favorite_df_no_high_pressure['Corrected Velocity'])

    # fav_area_scores_df = calculate_area_score(favorite_df_no_high_pressure, plot = False)
    # favorite_df_no_high_pressure = favorite_df_no_high_pressure.merge(fav_area_scores_df, on='Participant', how='inner')

    # ecdf_fn = empirical_cdf_fn(favorite_df_no_high_pressure['Corrected Velocity'])
    # ks_statistic_df = pd.DataFrame(columns=['Participant', 'KS Statistic', 'KS P-Value', 'EMD Score'])
    # for participant in favorite_df_no_high_pressure['Participant'].unique():
    #     participant_df = favorite_df_no_high_pressure[favorite_df_no_high_pressure['Participant'] == participant]
    #     participant_df_nhp = favorite_df_no_high_pressure[favorite_df_no_high_pressure['Participant'] == participant]
    #     participant_metrics = calculate_metrics(participant_df['Corrected Velocity'])
    #     skewness.append([participant,participant_metrics['skewness']])
    #     kurtosis.append([participant,participant_metrics['kurtosis']])
    #     ks_statistic, p_value = kstest(participant_df['Corrected Velocity'], ecdf_fn)
    #     emd_score = wasserstein_distance(participant_df['Corrected Velocity'], favorite_df_no_high_pressure['Corrected Velocity'])
    #     ks_statistic_df = pd.concat([ks_statistic_df, pd.DataFrame({'Participant': [participant], 'KS Statistic': [ks_statistic], 'KS P-Value': [p_value], 'EMD Score': [emd_score]})])
    #     # plot_ks_statistic(participant_df['Corrected Velocity'], favorite_df_no_high_pressure['Corrected Velocity'])
 
    
    # # merge ks statistic df with summary df
    # favorite_df_no_high_pressure = favorite_df_no_high_pressure.merge(ks_statistic_df, on='Participant', how='inner')

    # # rename Area Score columns to omit _x
    # favorite_df_no_high_pressure = favorite_df_no_high_pressure.rename(columns={'Area Score_x': 'Area Score', 'Log Area Score_x': 'Log Area Score'})
    # # run_regression(favorite_df_no_high_pressure, plot = True)
    


    # # plot velocities for each participant:
    # for participant in favorite_df_no_high_pressure['Participant'].unique():
    #     favorite_df_copy = favorite_df_no_high_pressure.copy()
    #     participant_df = favorite_df_copy[favorite_df_copy['Participant'] == participant]
    #     # Sort the data by 'Video':
    #     participant_df = participant_df.sort_values(by='Video')

    #     # iterate through videos in order and see if the video following the current video has the same pressure. Print the participant, capillary, location, video, pressure, and corrected velocity for each video that has the same pressure as the following video.
    #     for i in range(len(participant_df)-1):
    #         if (participant_df.iloc[i]['Pressure'] == participant_df.iloc[i+1]['Pressure']) and (participant_df.iloc[i]['Pressure'] != 0.2):
    #             # print(participant_df.iloc[i][['Participant', 'Capillary', 'Location', 'Video', 'Pressure', 'Corrected Velocity']]) # TODO: average out or drop videos
    #             pass
    #     # print(participant_df[['Participant', 'Capillary', 'Location', 'Video', 'Pressure', 'Corrected Velocity']])

        
    #     # Select columns to print:


    #     # plot_velocities(participant_df, write = False)
    #     # plot_densities_individual(summary_df, participant_df, participant)
    #     # plot_densities_pressure_individual(summary_df, participant_df, participant)
    #     # plot_cdf(favorite_df_no_high_pressure['Corrected Velocity'], 
    #     #          subsets=[favorite_df_no_high_pressure[favorite_df_no_high_pressure['Participant'] == participant]['Corrected Velocity']],
    #     #          labels=['Entire Dataset', participant], title=f'CDF Comparison of velocities for {participant}', 
    #     #          normalize = False)



    #     # Group the data by 'Capillary'
    #     grouped_df = participant_df.groupby('Capillary')
    #     # Get the unique capillary names
    #     capillaries = participant_df['Capillary'].unique()
    #     participant = participant_df['Participant'].unique()[0]
        


    #     # Plot each capillary's data in separate subplots
    #     for i, capillary in enumerate(capillaries):
    #         # print(f'Participant: {participant}, Capillary: {capillary}')
    #         capillary_data = grouped_df.get_group(capillary)
    #         capillary_data = capillary_data.copy()
    #         # decreases = capillary_data['Pressure'].diff() < 0
            
    #         # print(decreases)  

    #         # create column for "Up/down" in pressure by calling all videos after the maximum pressure 'down'
    #         capillary_data.loc[:, 'Up/Down'] = 'Up'
    #         max_pressure = capillary_data['Pressure'].max()
    #         max_index = capillary_data['Pressure'].idxmax()
    #         capillary_data.loc[max_index:, 'Up/Down'] = 'Down'

    #         # create function to fit a curve to the up and down data, respectively
    #         data_up = capillary_data[capillary_data['Up/Down'] == 'Up']
    #         data_down = capillary_data[capillary_data['Up/Down'] == 'Down']
    #         curve_up = plot_and_calculate_area(data_up, plot = False, normalize = False)
    #         curve_down = plot_and_calculate_area(data_down, plot = False, normalize = False)
    #         hysterisis = curve_up + curve_down
    #         # print(f'Participant: {participant}, Capillary: {capillary}, Hysterisis: {hysterisis}')
            
    #         # add hysterisis to the favorite_df
    #         favorite_df.loc[(favorite_df['Participant'] == participant) & (favorite_df['Capillary'] == capillary), 'Hysterisis'] = hysterisis
            
    # # # Plot scatter of age vs hysterisis 
    # # plt.figure(figsize=(10, 6))
    # # sns.scatterplot(x='Age', y='Hysterisis', data=favorite_df)
    # # plt.title('Hysterisis vs Age')
    # # plt.xlabel('Age')
    # # plt.ylabel('Hysterisis')
    # # plt.show()

    

           

                          
        


    return 0
    
if __name__ == '__main__':
    # to run the analysis code:
    main()
    # to make the summary_df_test file:
    # merge_vel_size(verbose=True)

    