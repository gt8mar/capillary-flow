import os
import platform
from matplotlib.font_manager import FontProperties


MANUSCRIPT_COLORS = {
    'control': '#1f77b4',
    'diabetes': '#ff7f0e',
    'hypertension': '#d62728',
    'age': '#1f77b4',
    'bp': '#2ca02c',
    'sex': '#674F92',
    'disease': '#00CED1',
}

MANUSCRIPT_PLOT_STYLE = {
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 8,
    'lines.linewidth': 0.5,
    'axes.linewidth': 0.5,
    'grid.linewidth': 0.3,
}

MANUSCRIPT_BOX_STYLE = {
    'figsize_single': (2.4, 2.0),
    'box_width': 0.6,
    'point_jitter_sd': 0.04,
    'point_alpha': 0.4,
    'point_size': 15,
    'point_color': 'black',
    'dpi': 300,
}


def apply_manuscript_plot_style():
    """Apply the shared manuscript matplotlib/seaborn style."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams.update(MANUSCRIPT_PLOT_STYLE)
    sns.set_style("whitegrid")


def save_manuscript_figure(fig, output_dir: str, filename: str, dpi: int = None,
                           png_subdir: bool = True, bbox_inches: str = 'tight'):
    """Save a manuscript figure as PDF and PNG using the shared save convention."""
    dpi = MANUSCRIPT_BOX_STYLE['dpi'] if dpi is None else dpi
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, f'{filename}.pdf')
    fig.savefig(pdf_path, dpi=dpi, bbox_inches=bbox_inches)

    png_dir = os.path.join(output_dir, 'png') if png_subdir else output_dir
    os.makedirs(png_dir, exist_ok=True)
    png_path = os.path.join(png_dir, f'{filename}.png')
    fig.savefig(png_path, dpi=dpi, bbox_inches=bbox_inches)
    return pdf_path, png_path


def get_paths():
    """Get system-specific paths based on hostname.
    
    Returns:
        dict: Dictionary containing paths for the current system
    """
    hostname = platform.node()
    
    computer_paths = {
        "LAPTOP-I5KTBOR3": {
            'cap_flow': 'C:\\Users\\gt8ma\\capillary-flow',
            'downloads': 'C:\\Users\\gt8ma\\Downloads'
        },
        "Quake-Blood": {
            'cap_flow': "C:\\Users\\gt8mar\\capillary-flow",
            'downloads': 'C:\\Users\\gt8mar\\Downloads'
        },
        "Clark-": {
            'cap_flow': "C:\\Users\\ejerison\\capillary-flow",
            'downloads': 'C:\\Users\\ejerison\\Downloads'
        },
        "Clark-qua-tr-5": {
            'cap_flow': "C:\\Users\\Luke\\capillary-flow",
            'downloads': 'C:\\Users\\Luke\\Downloads'
        },
        "Olric": {
            'cap_flow': "C:\\Users\\Marcus\\capillary-flow",
            'downloads': 'C:\\Users\\Marcus\\Downloads'
        },
    }
    
    default_paths = {
        'cap_flow': "/hpc/projects/capillary-flow",
        'downloads': "/home/downloads"
    }
    
    # Determine paths based on hostname
    paths = computer_paths.get(hostname, default_paths)
    
    # Add derived paths
    paths['frog_dir'] = os.path.join(paths['cap_flow'], "frog")
    paths['frog_analysis'] = os.path.join(paths['frog_dir'], "radial-analysis")
    paths['frog_results'] = os.path.join(paths['frog_dir'], "results")
    paths['frog_segmented'] = os.path.join(paths['frog_dir'], "segmented")
    
    # Create directories if they don't exist
    for path_name, path in paths.items():
        if path_name.endswith('_analysis') or path_name.endswith('_segmented') or path_name.endswith('_results'):
            os.makedirs(path, exist_ok=True)
    
    return paths

# Export paths directly for easier imports
PATHS = get_paths() 

# source sans loader:
def load_source_sans():
    """Load Source Sans font from downloads directory.
    
    Returns:
        FontProperties: Font properties for Source Sans
"""
 # Try to use Source Sans font if available, otherwise use default
    try:
        source_sans = FontProperties(fname=os.path.join(PATHS['downloads'], 'Source_Sans_3\\static\\SourceSans3-Regular.ttf'))
    except:
        print("Source Sans font not found, using default font")
        source_sans = None
    return source_sans
