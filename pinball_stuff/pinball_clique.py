import numpy as np
import stim


def syndrome_to_stim_idx(syndrome_idx, d):
    '''
    Converts index of x-basis syndrome to index within stim shot (which includes both x and z measurements)

    Parameters:
        syndrome_idx (int): Index of x-basis syndrome
        d (int): Distance of surface code

    Returns:
        stim_idx (int): Index within stim shot
    '''
    rows = d + 1
    cols = (d-1) // 2
    if syndrome_idx < cols:
        return syndrome_idx
    elif syndrome_idx >= rows * cols - cols:
        return rows * cols + syndrome_idx
    
    k1 = syndrome_idx - cols
    k2 = k1 // cols
    k3 = k2 * d
    k4 = k1 % cols
    return k3 + k4 * 2 + cols + 1

def clear_measurement_errors(prev_round, curr_round, num_rows, num_cols, d):
    for k in range(num_rows):
        for l in range(num_cols):
            inx = (k*num_cols) + l
            andval = curr_round[syndrome_to_stim_idx(inx, d)] & prev_round[syndrome_to_stim_idx(inx, d)]
            curr_round[syndrome_to_stim_idx(inx, d)] ^= andval
            prev_round[syndrome_to_stim_idx(inx, d)] ^= andval

def pinball_clique_late_meas(prev_syndrome, curr_syndrome, distance):
    #decoder decodes syndrome bits (this is the new version of clique (the pinball clique)
     # better for cmos?
     #the input to this is the syndrome array
     #should be replaced with your decoder
     #this is only tackling one type of error (i.e. x or z type, which is sufficient0
    """
    Generate an n x n output array based on the input array.

    Parameters:
        2 input_arrays (list of lists): The input array of size (n+1) * ((n-1)/2).

    Returns:
        output_array (list of lists): The generated output array of size n x n.
    """ 
    output_array = np.zeros(distance*distance, dtype=np.uint8)
    
    num_syndrome_rows = distance + 1
    num_syndrome_cols = (distance-1) // 2

    #data errors
    for trial in range(4):
        for i in range(num_syndrome_rows):
            for j in range(num_syndrome_cols):
                if(i%2==0): #only tackle odd rows
                    continue

                if(trial==0):
                    # top right
                    parity_row_index = i - 1
                    parity_col_index = j + 1 - i%2
                    data_row_index = i - 1
                    data_col_index = 2*(j+1) - i%2   
                if(trial==1):
                    # bottom right
                    parity_row_index = i + 1
                    parity_col_index = j + 1 - i%2
                    data_row_index = i
                    data_col_index = 2*(j+1) - i%2
                if(trial==2):
                    # bottom left
                    parity_row_index = i + 1
                    parity_col_index = j - i%2   
                    data_row_index = i
                    data_col_index =  2*(j+1) - i%2 - 1
                if(trial==3):
                    # top left
                    parity_row_index = i - 1
                    parity_col_index = j - i%2                 
                    data_row_index = i - 1
                    data_col_index = 2*(j+1) - i%2 - 1

                center_inx = (i*num_syndrome_cols) + j
                neighbor_inx = (parity_row_index * num_syndrome_cols) + parity_col_index
                data_inx = (data_row_index * distance) + data_col_index

                # Logic to avoid measurement errors being propagated to decoder
                value1 = curr_syndrome[syndrome_to_stim_idx(center_inx, distance)] # this has to exist

                if 0 <= parity_row_index < num_syndrome_rows and 0 <= parity_col_index < num_syndrome_cols:
                    value2 = curr_syndrome[syndrome_to_stim_idx(neighbor_inx, distance)]
                else:
                    value2 = -1 #due to clockwise this is okay

                if(0 <= data_row_index < distance and 0 <= data_col_index < distance and value2 != -1):
                    andval = value1 & value2
                    output_array[data_inx] ^= andval
                    curr_syndrome[syndrome_to_stim_idx(center_inx, distance)] ^= andval
                    if 0 <= parity_row_index < num_syndrome_rows and 0 <= parity_col_index < num_syndrome_cols:
                        curr_syndrome[syndrome_to_stim_idx(neighbor_inx, distance)] ^= andval   
                        
                # in general we would want to do this stage later because it only consumes one syndrome for an error 
                # it doesnt really matter here because this never steals from the above i.e. a 2 is always a 2.
                if(0 <= data_row_index < distance and 0 <= data_col_index < distance and value2 == -1):
                    value2=1
                    andval = value1 & value2
                    output_array[data_inx] ^= andval
                    curr_syndrome[syndrome_to_stim_idx(center_inx, distance)] ^= andval
                    if 0 <= parity_row_index < num_syndrome_rows and 0 <= parity_col_index < num_syndrome_cols:
                        curr_syndrome[syndrome_to_stim_idx(neighbor_inx, distance)] ^= andval                           

                ## These last lines effectively make this a 4-stage pipeline in order to avoid combinational loops
            
    ## additional code for last col - this could be combined with als pipe stage
    for i in range(num_syndrome_rows): 
        if(i%2==1): # only for even rows
            continue
        j = num_syndrome_cols - 1

        center_inx = i*num_syndrome_cols + j

        value1 = curr_syndrome[syndrome_to_stim_idx(center_inx, distance)]
        if(value1):
            #BR should always work
            data_col_index = distance-1
            data_row_index = i
            output_array[(data_row_index*distance) + data_col_index] ^= 1
            curr_syndrome[syndrome_to_stim_idx(center_inx, distance)] ^= 1

    # Measurement errors
    clear_measurement_errors(prev_syndrome, curr_syndrome, num_syndrome_rows, num_syndrome_cols, distance)
    
    return output_array

d = 5
noise = 0
num_shots = 1

circ = stim.Circuit.generated(
    "surface_code:rotated_memory_x",
    rounds=d,
    distance=d,
    after_clifford_depolarization=noise,
    after_reset_flip_probability=0,
    before_measure_flip_probability=0,
    before_round_data_depolarization=0)


shot = circ.compile_sampler().sample(num_shots)[0][0:-1].reshape(d + 1, d**2 - 1).astype(int)

initial_syndrome_x = []
for i in range(d+1):
    initial_syndrome_x.append(shot[0][syndrome_to_stim_idx(i, d)])
print("Initial syndrome:       ", initial_syndrome_x)

for i in range(d-1):
    prev_syndrome = shot[i]
    curr_syndrome = shot[i+1]
    corrections = pinball_clique_late_meas(prev_syndrome, curr_syndrome, d).reshape(d,d)
    print("Corrections after round " + str(i+1) + ": ", corrections)
    print("Syndrome after round " + str(i+1) + ": ", [curr_syndrome[syndrome_to_stim_idx(i, d)] for i in range(d+1)])

#print(corrections)

final_syndrome_x = []
for i in range(d+1):
    final_syndrome_x.append(curr_syndrome[syndrome_to_stim_idx(i, d)])
print("Final syndrome:         ", final_syndrome_x)


