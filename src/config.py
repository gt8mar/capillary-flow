import os
import platform

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
    }
    
    default_paths = {
        'cap_flow': "/hpc/projects/capillary-flow",
        'downloads': "/home/downloads"
    }
    
    # Determine paths based on hostname
    paths = computer_paths.get(hostname, default_paths)
    
    # Add derived paths
    paths['frog_analysis'] = os.path.join(paths['cap_flow'], "frog", "radial-analysis")
    
    # Create directories if they don't exist
    for path_name, path in paths.items():
        if path_name.endswith('_analysis'):  # Only create analysis directories
            os.makedirs(path, exist_ok=True)
    
    return paths 