import stim
import numpy as np
import pymatching
import math
from collections import Counter
import matplotlib.pyplot as plt

from stim_surface_code.memory import MemoryPatch
import os
from stim_surface_code.noise import NoiseParams


def max_length_error(det_coords, matched_idx, d):
    max_distance = 0
    max_coord1 = -1
    max_coord2 = -1
    for idx1, idx2 in matched_idx:
        i = det_coords[idx1]
        if idx2 == -1: # matched to boundary, distance must be nearest boundary
            dx = min(i[0]//2, (2*d+1 - i[0])//2)
            dy = min(i[1]//2, (2*d+1 - i[1])//2)
            dt = min(i[2]/1, d + 1 - i[2])
            distance = min(dx, dy, dt)
        else:
            j = det_coords[idx2]
            # Division by 2 for x,y reflects checkerboard distance in surface code lattice
            dx = abs(i[0] - j[0]) / 2
            dy = abs(i[1] - j[1]) / 2
            dt = abs(i[2] - j[2]) # difference between rounds
            distance = max(dx, dy, dt)

        if distance > max_distance:
            max_distance = distance
            max_coord1 = i
            max_coord2 = j if idx2 != -1 else -1

    return int(max_distance), max_coord1, max_coord2


def run_noise_analysis(distance, rounds, noise_model, noise_params, noise_briefs, num_shots):
    # initialize stim circuit based on specified noise model
    if noise_model == "phenomenological":
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            rounds=rounds,
            distance=distance,
            after_clifford_depolarization=0, # No 
            after_reset_flip_probability=0, # No reset flip
            before_measure_flip_probability=0.001,
            before_round_data_depolarization=noise_params)
    elif noise_model == "stim-based":
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            rounds=rounds,
            distance=distance,
            after_clifford_depolarization=noise_params,
            after_reset_flip_probability=0,
            before_measure_flip_probability=0.001,
            before_round_data_depolarization=0)
    elif noise_model == "realistic":
        patch = MemoryPatch(dx=distance, dz=distance, dm=rounds)
        noise_params.set_patch_err_vals(patch)
        circuit = patch.get_stim(observable_basis='Z')

    # retrieve detector coordinates map
    try:
      det_coords = {k: v for k, v in circuit.get_detector_coordinates().items()}
    except ValueError:
        print("Warning: Could not get detector coordinates. Max length calculation might fail.")
        det_coords = {}

    # prepare detector sampler and pymatching matcher
    sampler = circuit.compile_detector_sampler()
    dem = circuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching(dem)

    # sample the circuit for only detection events
    detection_events, _ = sampler.sample(shots=num_shots, separate_observables=True)

    # simulation loop
    count_map = Counter()
    for i in range(num_shots):

        detection_event = detection_events[i]
        # decode detection syndrome for MWPM detectors
        matched_dets = matcher.decode_to_matched_dets_array(detection_event)

        # find max length error chain for matched detectors
        max_len, p1, p2 = max_length_error(det_coords, matched_dets, distance)
        count_map[max_len] += 1

        # if (i + 1) % 1000 == 0:
        #     print(f"  Shot {i+1}/{num_shots}, Max Lengths: {sorted(count_map.items())}")

    print(f"Distance {distance}, Rounds {rounds}, Noise: {noise_briefs}")
    print(f"Max Length Counts: {count_map}")

    return count_map

def is_data(row, col):
    return row % 2 == 1 and col % 2 == 1

def construct_detector_grid(circuit: stim.Circuit, det_syndrome, d, rounds):
    qubit_coords = {k: (int(v[0]), int(v[1])) for k, v in circuit.get_final_qubit_coordinates().items()}
    det_coords = {k: (int(v[0]), int(v[1]), int(v[2])) for k, v in circuit.get_detector_coordinates().items()}
    grid = np.zeros((rounds+1, 2*d+1, 2*d+1), dtype=int)
    for round in range(rounds+1):
        for col in range(2*d+1):
            for row in range(2*d+1):
                if (row, col) not in qubit_coords.values():
                    grid[round, row, col] = 0
                elif (is_data(row, col)):
                    grid[round, row, col] = 1
                else:
                    grid[round, row, col] = 2
    for i, (row, col, round) in enumerate(det_coords.values()):
        grid[round, row, col] = det_syndrome[i] + 2
    return grid

def print_surface_code(grid, file):
    for round in range(grid.shape[0]):
        if round == 0:
            file.write("Initial Measurements\n")
        elif round != grid.shape[0]-1: 
            file.write(f"Between round {round-1} to {round}\n")
        else:
            file.write("Final State\n")
        file.write("    " + "".join(f" {col % 10} " for col in range(grid.shape[2])) + "\n")
        for row in range(grid.shape[1]):
            line = f"{row:3} "
            for col in range(grid.shape[2]):
                if grid[round, row, col] == 0:
                    line += "   "
                elif grid[round, row, col] == 1:
                    line += "|D|"
                elif grid[round, row, col] == 2:
                    line += "|_|"
                elif grid[round, row, col] == 3:
                    line += "|!|"
            file.write(line + "\n")
        file.write("\n")

def print_error_chain(grid, det_coord1, det_coord2, file):
    if det_coord1 == -1:
        file.write("No error chain detected.\n")
        return
    det_coord1 = (int(det_coord1[0]), int(det_coord1[1]), int(det_coord1[2]))

    for round in range(grid.shape[0]):
        if det_coord2 == -1:
            if round == det_coord1[2]:
                file.write("    " + "".join(f" {col % 10} " for col in range(grid.shape[2])) + "\n")
                for row in range(grid.shape[1]):
                    line = f"{row:3} "
                    for col in range(grid.shape[2]):
                        if grid[round, row, col] == 0:
                            line += "   "
                        elif grid[round, row, col] == 1:
                            line += "|D|"
                        elif (row, col, round) == det_coord1:
                            line += "|X|"
                        else:
                            line += "|_|"
                    file.write(line + "\n")
                file.write("\n")
        elif round >= det_coord1[2] and round <= det_coord2[2]:
            det_coord2 = (int(det_coord2[0]), int(det_coord2[1]), int(det_coord2[2]))
            file.write("    " + "".join(f" {col % 10} " for col in range(grid.shape[2])) + "\n")
            for row in range(grid.shape[1]):
                line = f"{row:3} "
                for col in range(grid.shape[2]):
                    if grid[round, row, col] == 0:
                        line += "   "
                    elif grid[round, row, col] == 1:
                        line += "|D|"
                    elif (row, col, round) == det_coord1 or (row, col, round) == det_coord2:
                        line += "|X|"
                    else:
                        line += "|_|"
                file.write(line + "\n")
            file.write("\n")


color_map_default = {
    0: 'lime',
    1: 'lightskyblue',
    2: 'gold',
    3: 'tomato'
}

def plot_max_length_error(all_results, distances, rounds, noise_model, noise_briefs, num_shots, color_map=color_map_default, output_dir="plots"):
    if color_map is None:
        color_map = color_map_default

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    num_noise_levels = len(noise_briefs)

    for i, d in enumerate(distances):
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(num_noise_levels)  # X-axis positions for noise levels

        # Prepare data for stacked bars
        percentages = []
        for j in range(num_noise_levels):
            count_map = all_results[i][j]
            grouped_count_map = Counter()
            for k, v in count_map.items():
                if k >= 3:
                    grouped_count_map["complex"] += v
                else:
                    grouped_count_map[k] += v
            level_percentages = [
                grouped_count_map.get(k, 0) / num_shots
                for k in range(3)
            ]
            level_percentages.append(
                grouped_count_map.get("complex", 0) / num_shots
            )
            percentages.append(level_percentages)

        # Plot stacked bars
        bottom = np.zeros(num_noise_levels)
        labels = ["0", "1", "2", "complex"]
        for k, label in enumerate(labels):
            heights = [percentages[j][k] * 100 for j in range(num_noise_levels)]  # Convert to percentage
            color_key = k % len(color_map)
            ax.bar(
                x,
                heights,
                width=0.5,
                label=label,
                bottom=bottom,
                color=color_map.get(color_key, "gray"),
            )
            bottom += heights

        ax.set_xlabel("Decoherence Noise (T1/T2)")
        ax.set_ylabel("Syndrome Error Percentage (%)")
        ax.set_xticks(x)
        ax.set_title(
            f"Effect of Decoherence Noise (T1/T2) on Error Syndrome Distribution \nDistance {d}, Rounds={rounds[i]}, Shots={num_shots}"
        ) 
        ax.set_xticklabels(noise_briefs, rotation=45, ha="right")
        ax.legend(title="Max Length", bbox_to_anchor=(1, 1), loc="upper right")
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        ax.set_ylim(0, 100) 

        plt.tight_layout()

        output_file = os.path.join(output_dir, f"decoherence_d{d}.png")
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")
        plt.close()


# === Debugging ===

# distance = 25
# rounds = 25 

# noise_params = 0.005

# circuit = stim.Circuit.generated(
#     "surface_code:rotated_memory_z",
#     rounds=rounds,
#     distance=distance,
#     after_clifford_depolarization=0,
#     after_reset_flip_probability=0,
#     before_measure_flip_probability=noise_params,
#     before_round_data_depolarization=noise_params
# )



# # noise_params = NoiseParams(baseline_error_means={'T1': 20e-6, 'T2': 30e-6, 'gate1_err': 1e-4, 'gate2_err': 5e-4, 'readout_err': 1e-3})

# # patch = MemoryPatch(dx=distance, dz=distance, dm=rounds)
# # noise_params.set_patch_err_vals(patch)
# # circuit = patch.get_stim(observable_basis='X')

# det_coords = {k: v for k, v in circuit.get_detector_coordinates().items()}
# sampler = circuit.compile_detector_sampler()
# dem = circuit.detector_error_model(decompose_errors=True)
# matcher = pymatching.Matching(dem)

# # Sample a single detection event
# detection_events, _ = sampler.sample(shots=1, separate_observables=True)
# detection_event = detection_events[0]
# matched_dets = matcher.decode_to_matched_dets_array(detection_event)

# # Find max length error
# max_len, det_coord1, det_coord2 = max_length_error(det_coords, matched_dets, distance)

# # Construct grid and print
# grid = construct_detector_grid(circuit, detection_event, distance, rounds)
# file = open("max_length_error_grid.txt", "w")

# print("Max Length Error:", max_len)
# print("First Detector Coordinate:", det_coord1, "Second Detector Coordinate:", det_coord2)
# print_error_chain(grid, det_coord1, det_coord2, file)
# file.close()



# === Configuration for error length distribution analysis ===

# distances = [3,7,15,31]
# rounds = distances
# num_shots = 100000

# # baseline_error_means_arr =  [{'T1': np.inf, 'T2': np.inf, 'gate1_err': 0.0001, 'gate2_err': 0, 'readout_err': 0.005},
# #                     {'T1': np.inf, 'T2': np.inf, 'gate1_err': 0.0001, 'gate2_err': 0.0001, 'readout_err': 0.005},
# #                     {'T1': np.inf, 'T2': np.inf, 'gate1_err': 0.0001, 'gate2_err': 0.0005, 'readout_err': 0.005},
# #                     {'T1': np.inf, 'T2': np.inf, 'gate1_err': 0.0001, 'gate2_err': 0.001, 'readout_err': 0.005}]

# baseline_error_means_arr =  [{'T1': 20e-6, 'T2': 30e-6, 'gate1_err': 0.0001, 'gate2_err': 0, 'readout_err': 0.005},
#                     {'T1': 20e-6, 'T2': 30e-6, 'gate1_err': 0.0001, 'gate2_err': 0.0001, 'readout_err': 0.005},
#                     {'T1': 20e-6, 'T2': 30e-6, 'gate1_err': 0.0001, 'gate2_err': 0.0005, 'readout_err': 0.005},
#                     {'T1': 20e-6, 'T2': 30e-6, 'gate1_err': 0.0001, 'gate2_err': 0.001, 'readout_err': 0.005}]

# realistic_noise_briefs = ["p2=0", "p2=0.0001", "p2=0.0005", "p2=0.001"]

# realistic_noise_configurations = [
#     NoiseParams(
#         baseline_error_means=baseline_error_means_arr[0]
#     ),
#     NoiseParams(
#         baseline_error_means=baseline_error_means_arr[1]
#     ),
#     NoiseParams(
#         baseline_error_means=baseline_error_means_arr[2]
#     ),
#     NoiseParams(
#         baseline_error_means=baseline_error_means_arr[3]
#     )
# ]

# phenomenological_noise_configurations = [0.001, 0.005, 0.01, 0.015]

# stim_based_noise_configurations = [0, 0.0001, 0.0005, 0.001]

# # PHENOMENOLOGICAL NOISE
# # noise_configurations = phenomenological_noise_configurations
# # noise_briefs = phenomenological_noise_configurations
# # noise_model = "phenomenological"


# # STIM-BASED NOISE
# noise_configurations = stim_based_noise_configurations
# noise_briefs = stim_based_noise_configurations
# noise_model = "stim-based"

# # REALISTIC NOISE
# # noise_configurations = realistic_noise_configurations 
# # noise_briefs = realistic_noise_briefs
# # noise_model = "realistic"


# all_results = [] 
# for d in distances:
#     print(f"--- Starting Distance {d} ---")
#     distance_results = []
#     for i, noise_params in enumerate(noise_configurations):
#         count_map = run_noise_analysis(
#             distance=d,
#             rounds=d,
#             noise_model = noise_model,
#             noise_params=noise_params,
#             noise_briefs=noise_briefs[i],
#             num_shots=num_shots
#         )
#         distance_results.append(count_map)
#     all_results.append(distance_results)
#     print(f"--- Finished Distance {d} ---")

# plot_max_length_error(all_results, distances, rounds, noise_model, noise_briefs, num_shots)

# print("Analysis complete.")

# circuit = stim.Circuit.generated(
#     "surface_code:rotated_memory_z",
#     rounds=2,
#     distance=15,
#     after_clifford_depolarization=0.01,
#     after_reset_flip_probability=0,
#     before_measure_flip_probability=0,
#     before_round_data_depolarization=0
# )

# detector_syndromes, _ = circuit.compile_detector_sampler().sample(shots=1, separate_observables=True)
# detector_syndrome = detector_syndromes[0]

# print_surface_code(
#     construct_detector_grid(circuit, detector_syndrome, 15, 2),
#     open("test.txt", "w")
# )


realistic_noise_configurations = [
    NoiseParams(
        baseline_error_means={'T1': np.inf, 'T2': np.inf, 'gate1_err': 0.0001, 'gate2_err': 0.0005, 'readout_err': 0.001}
    ),
    NoiseParams(
        baseline_error_means={'T1': 68e-6, 'T2': 89e-6, 'gate1_err': 0.0001, 'gate2_err': 0.0005, 'readout_err': 0.001}
    ),

]

distances = [15]
shots = 100000
p2_error = [0, 0.0001, 0.0005, 0.001]
all_count_map = []
for d in distances:
    print(f"--- Starting Distance {d} ---")
    distance_results = []
    # Run phenomenological noise analysis
    # distance_results.append(run_noise_analysis(distance=d, rounds=d, noise_model="phenomenological", noise_params=0.001, noise_briefs="phenomenological", num_shots=shots))
    for i, p2 in enumerate(realistic_noise_configurations):
        print(f"--- Starting Stim-Based, p2 = {p2}---")
        distance_results.append(run_noise_analysis(distance=d, rounds=d, noise_model="realistic", noise_params=p2, noise_briefs=f"p2={p2_error[i]}", num_shots=shots))
    all_count_map.append(distance_results)
    
plot_max_length_error(all_count_map, distances, distances, "stim-based", ["inf, inf", "68e-6, 89e-6"], shots)

