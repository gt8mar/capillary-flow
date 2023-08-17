import numpy as np
from collections import deque

def find_connected_components(input_array):
    def bfs(row, col, label):
        queue = deque([(row, col)])
        component_size = 0
        while queue:
            r, c = queue.popleft()
            # Check if the pixel is within bounds and is a valid part of the connected component
            if 0 <= r < input_array.shape[0] and 0 <= c < input_array.shape[1] and input_array[r, c] > 0 and labeled[r, c] == 0:
                labeled[r, c] = label
                component_size += 1
                # Extend the queue to explore neighboring pixels
                queue.extend([(r + dr, c + dc) for dr, dc in directions])
        return component_size
    
    # Define the possible directions for exploration (including diagonals)
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

    # Create an array to keep track of labeled components
    labeled = np.zeros_like(input_array, dtype=int)
    current_label = 1

    # Iterate over each pixel in the input array
    for i in range(input_array.shape[0]):
        for j in range(input_array.shape[1]):
            if input_array[i, j] > 0 and labeled[i, j] == 0:
                bfs(i, j, current_label)
                current_label += 1

    # Get unique labels assigned to valid connected components
    unique_labels = np.unique(labeled)
    unique_labels = unique_labels[unique_labels > 0]

    # Create a list to store individual 2D arrays representing connected components
    output_array = []
    for label in unique_labels:
        # Extract the component by replacing non-labeled pixels with zeros
        component = np.where(labeled == label, input_array, 0)
        output_array.append(component)

    result = []
    for component in output_array:
        nonzero_count = np.count_nonzero(component)
        if nonzero_count >= 2000:
            result.append(component)
    # Convert the list of 2D arrays to a 3D numpy array
    return np.array(result)


if __name__ == "__main__":
    find_connected_components()