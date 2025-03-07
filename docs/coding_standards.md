# Coding Standards Guide

## Plot Styling

### Standard Plot Configuration
```python
# Standard plot configuration
sns.set_style("whitegrid")
source_sans = FontProperties
(fname=os.path.join(downloads_path,'Source_Sans_3\\static\\SourceSans3-Regular.ttf'))

plt.rcParams.update({
    'pdf.fonttype': 42,  # For editable text in PDFs
    'ps.fonttype': 42,   # For editable text in PostScript
    'font.size': 7,
    'axes.labelsize': 7,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 5,
    'lines.linewidth': 0.5
})
```

### Standard Figure Dimensions
- Default figure size: `figsize=(2.4, 2.0)`
- Adjust as needed but maintain aspect ratio when possible

### Color Standards
```python
# Base colors for different categories
base_colors = {
    'default': '#1f77b4',
    'diabetes': '#ff7f0e',
    'hypertension': '#2ca02c',
    'heart_disease': '#d62728'
}

# For creating color palettes
def create_monochromatic_palette(base_color, n_colors=5):
    """Creates a monochromatic palette based on the given color."""
    # See implementation in plot_big.py
```

## File Organization

### Standard Import Order
```python
# Standard library imports
import os
import platform
from typing import Dict, Tuple, List

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import classification_report

# Local imports
from src.tools.parse_filename import parse_filename
```

### Path Management
```python
# Get the hostname and set paths
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
}
default_paths = {
    'cap_flow': "/hpc/projects/capillary-flow",
    'downloads': "/home/downloads"  # Adjust default downloads path as needed
}

# Determine paths based on hostname
paths = computer_paths.get(hostname, default_paths)
cap_flow_path = paths['cap_flow']
downloads_path = paths['downloads']

# Set up font
source_sans = FontProperties(fname=os.path.join(downloads_path, 'Source_Sans_3', 'static', 'SourceSans3-Regular.ttf'))

# Filetree folders
capillary-flow/
├── data
│   └── part00 # example participant
│   │   └── 240729 # exmaple date
|   │   │   └── loc01 #example location
|   |   │   │   ├── centerlines
|   |   |   │   │   └── coords
|   |   │   │   ├── segmented
|   |   |   │   │   └── hasty
|   |   │   │   ├── vids    
|   |   |   │   │   └── vid01 # example video
|   |   |   |   │   │   └── metadata
|   |   |   |   |   │   │   └── Results.csv
|   |   |   |   │   │   └── moco # Inside this folder (and mocoslice) are image frames which make a video
|   |   |   |   │   │   └── mocoslice # only in some shaky videos. 
├── docs
├── frog
│   └── 240729
│       ├── individual_caps
│       └── segmented
├── metadata
├── results
│   ├── decay_fits
│   │   └── participant_fits
│   ├── analytical
│   ├── kymographs
│   ├── segmented
│   │   └── proj_caps
│   ├── size
│   ├── stats
│   │   ├── comparisons
│   │   ├── classifier
│   │   │   ├── diabetes
│   │   │   ├── healthy
│   │   │   ├── heart_disease
│   │   │   └── hypertension
│   │   ├── pca
│   │   ├── umap
│   ├── total
│   ├── velocities
│   │   ├── too_fast
│   │   ├── too_slow
│   └── velocity_profiles
├── scripts
├── src
│   ├── __pycache__
│   ├── analysis
│   │   └── __pycache__
│   ├── simulation
│   └── tools
│       └── __pycache__
├── src.egg-info
└── tests

```

### Optical System constants
PIX_UM = 2.44 #1.74 for old camera

## Documentation Standards

### Function Docstrings
```python
def function_name(param1: type, param2: type, optional_param: type = default) -> return_type:
    """Brief one-line description of function purpose.

    Longer description if needed. Explain the function's purpose, behavior,
    and any important implementation details.

    Args:
        param1: Description of first parameter
        param2: Description of second parameter
        optional_param: Description of optional parameter, include default value

    Returns:
        Description of return value(s)

    Raises:
        ExceptionType: Description of when/why this exception is raised

    Example:
        >>> result = function_name(1, "test", optional_param=True)
        >>> print(result)
        42
    """
```

### Plot Function Documentation
```python
def plot_something(df: pd.DataFrame, variable: str = 'Age', 
                  log: bool = False, write: bool = False) -> int:
    """Creates a standardized plot comparing groups based on the specified variable.

    Creates and optionally saves a plot showing the relationship between groups
    defined by the variable parameter. Handles data preprocessing, plotting,
    and statistical analysis.

    Args:
        df: DataFrame containing the data to plot
        variable: Column name to group by ('Age', 'SYS_BP', 'Sex', etc.)
        log: Whether to use log scale for y-axis
        write: Whether to save the plot to disk

    Returns:
        0 if successful, 1 if error occurred

    Example:
        >>> plot_something(data, variable='Age', log=True, write=True)
        0
    """
```

### Class Documentation
```python
class ClassName:
    """Brief description of the class's purpose.

    Detailed description of the class, its behavior, and important notes
    about usage. Include any key assumptions or limitations.

    Attributes:
        attr1: Description of first attribute
        attr2: Description of second attribute
    """

    def __init__(self, param1: type, param2: type = default):
        """Initialize the class.

        Args:
            param1: Description of first parameter
            param2: Description of second parameter, note default value
        """
```

### Module Documentation
```python
"""Module name and brief description.

Detailed description of the module's purpose, key functionality,
and any important notes about usage.

Example:
    >>> from module_name import function_name
    >>> result = function_name(param1, param2)

Attributes:
    CONSTANT_NAME: Description of module-level constants
    
Functions:
    function_name: Brief description of key functions
"""
```

### Comment Style Guidelines

1. Inline Comments
```python
x = 1  # Brief explanation when needed
```

2. Block Comments
```python
# Section header
# -------------
# Detailed explanation of complex code section
# Include key assumptions and edge cases
```

3. TODO Comments
```python
# TODO(username): Brief description of needed changes
# TODO(username): Add support for new feature (Issue #123)
```

4. Warning Comments
```python
# WARNING: Explain potential pitfalls or important caveats
# FIXME: Document known issues that need addressing
```

### Documentation Best Practices

1. Keep docstrings concise but complete
2. Use type hints for all function parameters and return values
3. Include examples for non-obvious usage
4. Document exceptions and edge cases
5. Keep comments up to date with code changes
6. Use proper spelling and grammar
7. Follow Google Python Style Guide for docstring formatting
```

## Data Standards

### Standard DataFrame Columns
Below are the typical columns found in our analysis DataFrames, along with example values:

```python
standard_columns = {
    # Participant Information
    'Participant': 'part99',          # Participant ID
    'Age': 54.0,                      # Age in years
    'Sex': 'F',                       # Sex (M/F)
    'Birthday': '19681220',           # YYYYMMDD format
    'Height': float,                  # Height (optional)
    'Weight': float,                  # Weight (optional)
    
    # Health Conditions (Boolean)
    'Hypertension': False,
    'Diabetes': False,
    'Raynauds': False,
    'SickleCell': False,
    'SCTrait': False,
    'HeartDisease': False,
    'Other': False,
    
    # Vital Signs
    'BP': '137/70',                   # Blood pressure reading
    'SYS_BP': 137,                    # Systolic blood pressure
    'DIA_BP': 70,                     # Diastolic blood pressure
    'Pulse': 60,                      # Heart rate
    'PulseOx': float,                 # Pulse oximetry (optional)
    
    # Video/Capture Information
    'Date': '230414',                 # YYMMDD format
    'Location': 'loc01',              # Location ID
    'Video': 'vid01',                 # Video ID
    'VideoID': 1,                     # Numeric video identifier
    'SET': 'set01',                   # Set identifier
    'Finger': 'LMid',                 # Finger position
    'Exposure': 2813,                 # Camera exposure
    'FPS': 227.8,                     # Frames per second
    'Pressure': 0.2,                  # Applied pressure in psi
    
    # Capillary Measurements
    'Capillary': '00a',              # Capillary identifier
    'Centerline Length': 374.0,       # Length in pixels
    'Area': 11758.0,                  # Area in pixels
    'Diameter': 31.43850267,          # Diameter in pixels
    
    # Velocity Measurements
    'Velocity': 329.194,              # Raw velocity
    'Corrected Velocity': 329.194,    # Corrected velocity
    'Manual Velocity': float,         # Manual velocity measurement
    'Video_Median_Velocity': 358.33755,
    'Video_Mean_Velocity': 351.1815,
    'Video_Std_Velocity': 90.44684553,
    'Video_Max_Velocity': 450.4171,
    'Video_Min_Velocity': 237.6338,
    
    # Statistical Measures
    'Video_Skew_Velocity': -0.233847302,
    'Video_Kurt_Velocity': -1.238906138,
    'Age-Score': 47.34778358,
    'Log Age-Score': 0.181144038,
    'KS Statistic': 0.182789544,
    'KS P-Value': 8.33e-09,
    'EMD Score': float,               # Earth Mover's Distance score
    
    # Quality Control
    'Correct': 't',                   # Quality check passed
    'Correct2': str,                  # Secondary quality check
    'Zero': 'f',                      # Zero velocity flag
    'Max': 'f',                       # Maximum velocity flag
    'Drop': str,                      # Drop flag
    
    # Grouping Variables
    'Age_Group': 'Below 50',          # Age grouping
    'Sex_Group': 'F',                 # Sex grouping
    'BP_Group': '<=120',              # Blood pressure grouping
    
    # Notes and Additional Information
    'Notes': str,                     # General notes
    'Notes_x': str,                   # Additional notes
    'Notes2': str,                    # Secondary notes
    'Notes_y': 'dropping frames',     # Video-specific notes
}
```

When working with these DataFrames:
1. Maintain consistent column names across analysis scripts
2. Use appropriate data types (float for numerical values, boolean for flags)
3. Handle missing values (typically represented as `nan`) appropriately
4. Document any new columns added to the standard set
5. Maintain the integrity of boolean health condition flags