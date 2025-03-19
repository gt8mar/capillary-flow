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
        "Clark-": {
            'cap_flow': "C:\\Users\\ejerison\\capillary-flow",
            'downloads': 'C:\\Users\\ejerison\\Downloads'
        }
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
    paths['frog_segmented'] = os.path.join(paths['frog_dir'], "segmented")
    
    # Create directories if they don't exist
    for path_name, path in paths.items():
        if path_name.endswith('_analysis') or path_name.endswith('_segmented'):
            os.makedirs(path, exist_ok=True)
    
    return paths

# Export paths directly for easier imports
PATHS = get_paths() 