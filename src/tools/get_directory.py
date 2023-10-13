import os

def get_directory_at_level(path, level, only_dir=True):
    for _ in range(level):
        path = os.path.dirname(path)
    if only_dir:
        return os.path.basename(path)
    else:
        return path