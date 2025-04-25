import numpy as np
import pymatching

# Helper functions
import utils

from sim_clique import pinball_clique_late_meas

from copy import deepcopy

# Configuration info for running simulations - passed to the run_simulation function
class SimConfig():
    def __init__(self, distance, error_rate, batch_size, num_batches, thread_id):
        self.distance = distance # Code distance
        self.error_rate = error_rate # Physical error rate
        self.batch_size = batch_size # How many syndrome rounds per batch (typically d)
        self.num_batches = num_batches # Number of batches of syndromes to simulate over
        self.thread_id = thread_id # Useful for debugging

# phenomenological simulation
def run_simulation(sim_config): 
    d = sim_config.distance
    e = sim_config.error_rate
    batch_size = sim_config.batch_size
    num_rounds = sim_config.num_batches * batch_size
    
    # Syndrome array dimensions
    num_syndrome_rows = d + 1
    num_syndrome_cols = (d - 1) // 2

    # Instantiate MWPM decoder
    check_matrix = utils.generate_X_parity_check_matrix(d)
    mwpm = pymatching.Matching(check_matrix, repetitions=d)

    # np.random.seed(sim_config.thread_id) # NOTE: uncomment this for deterministic outputs
    # Generate data errors and associated syndromes
    (data_errors, syndromes) = utils.generate_decoder_inputs(d, e, check_matrix, num_rounds)
    original_syndromes = deepcopy(syndromes) # Useful for debugging purposes

    # Simulation output statistics
    num_complex = 0 # Number of complex syndrome batches
    num_all_zeros = 0 # Number of syndrome batches that had no active syndromes

    mwpm_errors = 0 # Number of syndrome batches MWPM fails to correct successfully
    pinball_errors = 0 # Number of syndrome batches Pinball fails to correct successfully
    pinball_only_errors = 0 # Number of syndrome batches ONLY Pinball fails to correct successfully
                            # (subset of pinball_errors)

    # Bins for categorizing failure scenarios
    mwpm_only_failures = [] # List of batches which ONLY MWPM failed to correct
    pinball_only_failures = [] # List of batches which ONLY Pinball failed to correct
    both_failures = [] # List of batches which BOTH MWPM AND Pinball failed to correct

    prev_syndrome = np.zeros(num_syndrome_rows*num_syndrome_cols, dtype=np.uint8)

    # Iterate over batches of errors/syndromes
    for i in range(0, num_rounds, batch_size):
        
        # Pull out a batch of syndrome/error rounds
        error_batch = data_errors[i:i+batch_size]
        syndrome_batch = syndromes[i:i+batch_size]

        # Pinball will override syndromes, so keep a copy for MWPM to work with
        mwpm_syndrome_batch = deepcopy(syndrome_batch)
        
        # If the batch is all zeros, both Pinball and MWPM will definitely succeed, so skip
        if not np.any(syndrome_batch):
            prev_syndrome = np.zeros(num_syndrome_rows*num_syndrome_cols, dtype=np.uint8)
            num_all_zeros += 1
            continue

        # Tracks if any rounds in batch were complex
        batch_complex = False

        # Per-round corrections produced by Pinball
        pinball_correction = np.zeros(d*d, dtype=np.uint8)

        # Attempt correction with Pinball
        for j, syndrome in enumerate(syndrome_batch):
            # Handle data errors in the current round and measurement errors between
            # current and prev. round
            # NOTE: this function clears bits in syndrome inputs
            correction = pinball_clique_late_meas(prev_syndrome, syndrome, d, j)
            pinball_correction ^= correction

            prev_syndrome = syndrome
        
        # A batch of rounds is complex if any syndrome bits remain uncleared
        batch_complex = np.any(syndrome_batch)

        # If complex, Pinball will have deferred to 2nd level decoder, so don't
        # analyze correction results
        if batch_complex:
            num_complex += 1

        # In cases where Pinball was used, compare accuracy with 2nd level decoder
        else:
            mwpm_correction = mwpm.decode(mwpm_syndrome_batch.T)

            # Flatten rounds of errors and combine with corrections
            flat_error = np.bitwise_xor.reduce(error_batch)
            pinball_xor = pinball_correction ^ flat_error
            mwpm_xor = mwpm_correction ^ flat_error

            # Verify the corrections from both decoders
            pinball_fail = utils.verify_decoder_corrections(pinball_xor, check_matrix)
            mwpm_fail = utils.verify_decoder_corrections(mwpm_xor, check_matrix)
                
            # Count up and bin the failures
            if pinball_fail:
                if not mwpm_fail:
                    pinball_only_errors += 1
                pinball_errors += 1
            if mwpm_fail:
                mwpm_errors += 1

            if pinball_fail and mwpm_fail:
                both_failures.append(i // batch_size)
            elif pinball_fail:
                pinball_only_failures.append(i // batch_size)
            elif mwpm_fail:
                mwpm_only_failures.append(i // batch_size)
        
    return (pinball_errors, mwpm_errors, pinball_only_errors, num_complex, num_all_zeros, both_failures, pinball_only_failures, mwpm_only_failures)



def run_stim_simulation(sim_config, syndromes, num_rounds, num_shots):
    d = sim_config.distance
    batch_size = sim_config.batch_size
    
    # Syndrome array dimensions
    num_syndrome_rows = d + 1
    num_syndrome_cols = d**2-1

    # Simulation output statistics
    num_complex = 0 # Number of complex syndrome batches
    num_all_zeros = 0 # Number of syndrome batches that had no active syndromes

    prev_syndrome = np.zeros(num_syndrome_rows*num_syndrome_cols, dtype=np.uint8) # TODO: something wrong here

    # Iterate over batches of errors/syndromes
    for shot in range(num_shots):
        
        # Pull out a batch of syndrome/error rounds
        syndrome_batch = syndromes[shot].reshape(num_rounds, d**2-1)
        
        # If the batch is all zeros, both Pinball and MWPM will definitely succeed, so skip
        if not np.any(syndrome_batch):
            prev_syndrome = np.zeros(num_syndrome_rows*num_syndrome_cols, dtype=np.uint8)
            num_all_zeros += 1
            continue

        # Tracks if any rounds in batch were complex
        batch_complex = False

        # Per-round corrections produced by Pinball
        pinball_correction = np.zeros(d*d, dtype=np.uint8)

        # Attempt correction with Pinball
        for j, syndrome in enumerate(syndrome_batch):
            # Handle data errors in the current round and measurement errors between
            # current and prev. round
            # NOTE: this function clears bits in syndrome inputs
            correction = pinball_clique_late_meas(prev_syndrome, syndrome, d, j)
            pinball_correction ^= correction

            prev_syndrome = syndrome
        
        # A batch of rounds is complex if any syndrome bits remain uncleared
        batch_complex = np.any(syndrome_batch)

        # If complex, Pinball will have deferred to 2nd level decoder, so don't
        # analyze correction results
        if batch_complex:
            num_complex += 1

    return num_complex

import stim

# d=9
# sim = SimConfig(distance=d, error_rate=0, batch_size=d*(d**2-1), num_batches=1000, thread_id=0)


# circ = stim.Circuit.generated(
#     "surface_code:rotated_memory_x",
#     rounds=d,
#     distance=d,
#     after_clifford_depolarization=0.0005,
#     after_reset_flip_probability=0,
#     before_measure_flip_probability=0.001,
#     before_round_data_depolarization=0)

# syndrome_bools, _ = circ.compile_detector_sampler().sample(shots=sim.num_batches, separate_observables=True)

# syndromes = syndrome_bools.astype(np.uint8)
# print(run_stim_simulation(sim, syndromes, sim.num_batches))





def run_coverage_simulation(distances, error_rates, num_shots):
    complex_count_map = []
    for p in phen_errors:
        print(f"Starting Noise Level {p}")
        complex_errors_d = []
        for d in distances:
            num_rounds = d
            print(f"Starting Distance {d}")
            sim = SimConfig(distance=d, error_rate=p, batch_size=num_rounds*(d**2-1), num_batches=num_shots, thread_id=0)
            circ = stim.Circuit.generated(
                "surface_code:rotated_memory_x",
                rounds=num_rounds,
                distance=d,
                after_clifford_depolarization=0,
                after_reset_flip_probability=0,
                before_measure_flip_probability=p,
                before_round_data_depolarization=p)
            syndrome_bools, _ = circ.compile_detector_sampler().sample(shots=num_shots, separate_observables=True)
            syndromes = syndrome_bools.astype(np.uint8)
            complex_count = run_stim_simulation(sim, syndromes, num_rounds, num_shots)
            complex_errors_d.append(complex_count)
            print(f"Number of Complex Errors: {complex_count}")
        complex_count_map.append(complex_errors_d)

    percent_on_chip_map = [[100 - (count / num_shots) * 100 for count in counts] for counts in complex_count_map]
    return percent_on_chip_map

import matplotlib.pyplot as plt

distances = [3,5,7,9,11,13,15,17,19,21]
phen_errors = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
num_shots = 1000
percent_on_chip_map = run_coverage_simulation(distances, phen_errors, num_shots)

# Plot the data
for i, p in enumerate(phen_errors):
    plt.plot(distances, percent_on_chip_map[i], label=f"p={p}", markersize=8, linewidth=2)

plt.xlabel("Distance", fontsize=14)
plt.ylabel("Percent On-Chip Decode (%)", fontsize=10)
plt.title("Percent On-Chip Decode vs Distance", fontsize=12)
plt.legend(fontsize=10)  # Make the legend smaller
plt.grid(True)

# Set x-axis ticks to be at the distances
plt.xticks(distances, fontsize=12)
plt.yticks(fontsize=12)

# Save the plot to a file
plt.savefig("circuit_level_coverage.png")


        
        


