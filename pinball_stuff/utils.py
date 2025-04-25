import random
import copy
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from enum import Enum

class Decoder(Enum):
    MWPM = 0
    SLIDING_WINDOW = 1
    BASELINE = 2
    PINBALL = 3

# Generates the differences between successive values of the data qubits
# E.g., A "1" indicates that the value of the data qubit changed (0 -> 1 or 1 -> 0)
#       A "0" indicates that the value of the data qubit did not change      

'''
    Generate a square 2D array of random data errors based on a given probability

    Inputs:
        n (int): the dimension of the square 2D array
        probability (float): probability of an element in the output array being 1, between 0.0 and 1.0
'''  
def generate_random_array(n, probability):
    # random_array = np.random.rand(n,n) < probability
    random_array = np.random.rand(n*n) < probability

    return random_array.astype(np.uint8)

'''
    For a given array of data qubit errors, generate an array
    with the parity check (syndrome) values.

    UPDATE: not used anymore, replaced with multiplication of data_array
    with parity check matrix (faster). Keeping here for potential compatibility
    issues
'''
def generate_syndrome_array(data_array):
    d = len(data_array)
    num_rows = d+1
    num_cols = (d-1)//2

    # Pad the top and bottom of the data array with 0s
    padded = np.pad(data_array, pad_width=((1,1),(0,0)), mode="constant", constant_values=0)

    output_array = np.zeros((num_rows, num_cols), dtype=np.uint8)
    
    # Pre-compute possible column offsets
    col_indices = 2*np.arange(num_cols)
    offsets_even = col_indices
    offsets_odd = col_indices + 1
    
    for i in range(num_rows):
        offsets = offsets_odd if (i % 2 == 1) else offsets_even

        top_left = padded[i, offsets]
        top_right = padded[i, offsets+1]
        bottom_left = padded[i+1, offsets]
        bottom_right = padded[i+1, offsets+1]

        output_array[i] = top_left ^ top_right ^ bottom_left ^ bottom_right

    return output_array

# Generates syndrome inputs for decoder
def generate_decoder_inputs(distance, error_rate, check_matrix, num_trials):
    num_syndrome_rows = distance + 1
    num_syndrome_cols = (distance - 1) // 2

    # array of input data qubits for each trial
    data_errors = np.zeros((num_trials, distance*distance), dtype=np.uint8)
    # array of parity check measurements (syndromes) for each trial
    syndromes = np.zeros((num_trials, num_syndrome_rows*num_syndrome_cols), 
                         dtype=np.uint8)
    
    # Generate data errors and associated syndromes
    for i in range(num_trials):
        data_errors[i] = generate_random_array(distance, error_rate)
        syndromes[i] = check_matrix @ data_errors[i] % 2

    # Now introduce measurement errors
    for i in range(num_trials-1):
        current_syndrome = syndromes[i]
        next_syndrome = syndromes[i+1]

        for j in range(num_syndrome_rows):
            for k in range(num_syndrome_cols):
                flip = 1 if np.random.random() < error_rate else 0
                if(flip):
                    # flip here corresponds to the actual measurement error
                    current_syndrome[j*num_syndrome_cols + k] ^= 1
                    # flip here corresponds to the next measurement returning the correct value again 
                    next_syndrome[j*num_syndrome_cols + k] ^= 1
    
    return data_errors, syndromes

'''
    Compute the element-wise XOR of two 2D arrays
'''
def array_xor(array1, array2):
    n = len(array1)
    xor_array = []

    for i in range(n):
        row = []
        for j in range(n):
            # Perform XOR operation on corresponding elements of the two arrays
            xor_value = array1[i][j] ^ array2[i][j]
            row.append(xor_value)
        xor_array.append(row)

    return xor_array

'''
    Utilities for checking correctness of decoder corrections.
    Checks if the resulting error pattern is:
        1. A valid correction (all syndromes are cleared)
        2. A logical error
        3. A stabilizer

    Inputs:
        corrections_arrray: length d^2 array of corrections to apply to data qubits

    Returns:
        2, if decoder corrections were invalid (uncleared syndromes left)
        1, if decoder corrections produced a logical error
        0, otherwise
'''
def verify_decoder_corrections(corrections_array, check_matrix):
    # Dimension, d, of the data qubit array
    d = int(sqrt(len(corrections_array)))

    # Returns 0 if all syndromes are cleared, 1 otherwise
    def check_all_syndromes_clear():
        # Iterate over rows and columns of ancillas
        for row in range(d+1):
            for col in range((d-1)//2):
                # Even-indexed row
                if (row % 2) == 0:
                    top_left = d*(row-1) + 1 + 2*col
                    top_right = top_left + 1
                    bottom_left = d*row + 1 + 2*col
                    bottom_right = bottom_left + 1
                # Odd-indexed row
                else:
                    top_left = d*(row-1) + 2*col
                    top_right = top_left + 1
                    bottom_left = d*row + 2*col
                    bottom_right = bottom_left + 1

                parity = 0
                if top_left >= 0:
                    parity ^= corrections_array[top_left]
                if top_right >= 0:
                    parity ^= corrections_array[top_right]
                if bottom_left < d*d:
                    parity ^= corrections_array[bottom_left]
                if bottom_right < d*d:
                    parity ^= corrections_array[bottom_right]
                
                # This ancilla would not have been cleared,
                # return early
                if (parity):
                    return 1
        
        # All ancillas passed
        return 0
    
    # Depth-first search implementation, used to construct
    # correction chains
    visited = [False for _ in range(len(corrections_array))]
    def dfs(row, col):
        stack = [(row, col)]
        chain = []
        while stack:
            r, c = stack.pop()
            if not visited[row * d + col]:
                visited[row*d + col] = True
                chain.append((r, c))
                directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < d and 0 <= nc < d and corrections_array[nr*d + nc] == 1 and not visited[nr*d + nc]:
                        stack.append((nr, nc))
        return chain
    
    # Builds up chains of the continguous corrections
    def trace_corrections():
        chains = []
        for i in range(d):
            for j in range(d):
                if corrections_array[i*d + j] == 1 and not visited[i*d + j]:
                    chain = dfs(i, j)
                    chains.append(chain)
        
        return chains
    
    # Generate a pair of vertices representing the endpoints
    # of the given chain
    def get_chain_endpoints(chain):
        endpoints = []
        for r, c in chain:
            # Check how many neighbors (top left, top right, ...) are also in the chain
            neighbors = sum((r + dr, c + dc) in chain for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)])
            
            # If there is only one neighbor, it must be an endpoint of the chain
            if neighbors == 1:
                endpoints.append((r, c))

        return endpoints

    # Check if a given chain of corrections forms a logical error
    def check_logical_error(chain):
        # data = np.array(corrections_array)
        
        data = np.reshape(corrections_array, (d, d))

        # We're decoding Z errors so we need to go over the columns of
        # the data qubit array
        transpose = np.transpose(data).tolist()

        # If any column has an odd number of corrections in it,
        # we've formed a logical error
        for row in transpose:
            if row.count(1) % 2:
                return 1
            
        return 0

    # == CHECKING VALIDITY OF CORRECTIONS ==
    
    # First, we will check that all syndromes are cleared
    # (i.e., the corrections have formed some kind of loop(s))
    if check_all_syndromes_clear():
        return 2 # indicate invalid decode
    
    # Next, we will collect all chains of corrections and check
    # whether any of them form a logical error
    chains = trace_corrections()
    for chain in chains:
        if check_logical_error(chain):
            return 1

    # If we've gotten this far, the corrections successfully formed
    # a stabilizer or a product of stabilizers
    return 0

def generate_X_parity_check_matrix(d):
    """
    Generate the parity check matrix for the X stabilizers)
    of a distance d rotated surface code
    
    Parameters:
        d (int): Distance of the rotated surface code (must be an odd integer)
    
    Returns:
        np.ndarray: A binary matrix representing the X stabilizer parity check matrix.

        
    Example: Distance-3 X stabilizer check matrix
    
    check_matrix = np.array([
        [0, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 0]
    ])
    """
    if d % 2 == 0:
        raise ValueError("Distance d must be an odd integer.")
    
    n_qubits = d**2  # Total number of qubits (on edges of the lattice)
    n_stabilizers = ((d+1)*(d-1))//2  # Number of vertex stabilizers
    
    def get_qubit_index(row, col):
        # return col
        return row * d + col
    
    # Initialize the stabilizer matrix
    H_X = np.zeros((n_stabilizers, n_qubits), dtype=np.uint8)
    
    # Iterate over the stabilizers
    for row in range(d+1):
        for col in range((d-1)//2):
            stabilizer_idx = (row * (d-1)//2) + col
            if row == 0:
                H_X[stabilizer_idx, get_qubit_index(0, 2*col + 1)] = 1
                H_X[stabilizer_idx, get_qubit_index(0, 2*(col+1))] = 1
            elif row == d:
                H_X[stabilizer_idx, get_qubit_index(d-1, 2*col)] = 1
                H_X[stabilizer_idx, get_qubit_index(d-1, 2*col + 1)] = 1
            # Odd row
            elif row % 2:
                H_X[stabilizer_idx, get_qubit_index(row-1, 2*col)] = 1
                H_X[stabilizer_idx, get_qubit_index(row-1, 2*col + 1)] = 1
                H_X[stabilizer_idx, get_qubit_index(row, 2*col)] = 1
                H_X[stabilizer_idx, get_qubit_index(row, 2*col + 1)] = 1
            # Even row
            else:
                H_X[stabilizer_idx, get_qubit_index(row - 1, 2*col + 1)] = 1
                H_X[stabilizer_idx, get_qubit_index(row - 1, 2*(col+1))] = 1
                H_X[stabilizer_idx, get_qubit_index(row, 2*col + 1)] = 1
                H_X[stabilizer_idx, get_qubit_index(row, 2*(col+1))] = 1

    return H_X

'''
    Utility function to graph decoder accuracy (of the decodes that Clique handled, the % it corrected successfully)

    Inputs:
        distances: code distances that were simulated
        error_rates: physical error rates that were simulated
        num_trials: number of simulated trials per code distance
        accuracies_per_distance: 2D array containing, per code distance, the decoding accuracy for each physical error rate
        color_series: for plot formatting, use a different color per code distance
'''
def graph_decoding_accuracy(distances, error_rates, num_trials, coverages_per_distance, success_counts_per_distance, color_series):
    for d in range(len(distances)):
        # For the given code distance, number of trials that Clique decoded successfully
        # at each physical error rate
        success_counts = success_counts_per_distance[d]
        # For the given code distance, the number of trials that Clique handled at each
        # physical error rate
        coverages = coverages_per_distance[d]

        accuracies = []
        for j in range(len(success_counts)):
            # Calculate decoding accuracy as percentage of successes over the 
            # trials that Clique covered
            accuracies.append(success_counts[j] * 100 / coverages[j])
        plt.plot(error_rates, accuracies, c=color_series[d], marker=".")

    plt.xlabel("Physical Error Rate")
    plt.ylabel("Decoding Accuracy")

'''
    Utility function to graph decoder coverage (% of decodes that Clique was able to handle)

    Inputs:
        distances: code distances that were simulated
        error_rates: physical error rates that were simulated
        num_trials: number of simulated trials per code distance
        coverage_per_distance: 2D array containing, per code distance, the coverage for each physical error rate
        color_series: for plot formatting, use a different color per code distance
'''
def graph_coverage(distances, error_rates, num_trials, coverages_per_distance, color_series):
    for i in range(len(distances)):
        # For the given code distance, the number of trials that Clique handled at each
        # physical error rate
        coverages = coverages_per_distance[i]
        ratios = []
        for c in coverages:
            # Calculate coverage as percentage of total trials
            ratios.append(c*100/num_trials)
        plt.plot(error_rates, ratios, c=color_series[i], marker=".")
    plt.ylabel("Coverage")
    plt.xlabel("Physical Error Rate")
    plt.savefig("circuit_level_coverage.png")