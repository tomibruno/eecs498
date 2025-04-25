import copy
import utils
import matplotlib.pyplot as plt
import argparse
import numpy as np

def original_clique(curr_syndrome, prev_syndrome, distance):
     #decoder decodes syndrome bits (this is a original but updated version of clique)
     #the input to this is 2 syndrome arrays 
    # the first is the current and the second is the old one.
    #this is done to combine two rounds of measurements as is discussed in the paper
     #this is only tackling one type of error (i.e. x or z type, which is sufficient0
    """
    Generate an n x n output array based on the input array.

    Parameters:
        2 input_arrays (list of lists): The input array of size (n+1) * ((n-1)/2).

    Returns:
        output_array (list of lists): The generated output array of size n x n.
    """

    output_array = np.zeros(distance*distance, dtype=np.uint8)
    num_syndrome_rows = distance+1
    num_syndrome_cols = (distance-1) // 2
    
    #data errors
    for i in range(num_syndrome_rows):
        for j in range(num_syndrome_cols):
            #if(i%2==0 and j != len(input_array[0])): #only tackle odd rows, unless it is last col
            #    continue
            # top right
            tr_parity_row_index = i - 1
            tr_parity_col_index = j + 1 - i%2
            tr_data_row_index = i - 1
            tr_data_col_index = 2*(j+1) - i%2   
            # bottom right
            br_parity_row_index = i + 1
            br_parity_col_index = j + 1 - i%2
            br_data_row_index = i
            br_data_col_index = 2*(j+1) - i%2
            # bottom left
            bl_parity_row_index = i + 1
            bl_parity_col_index = j - i%2   
            bl_data_row_index = i
            bl_data_col_index =  2*(j+1) - i%2 - 1
            # top left
            tl_parity_row_index = i - 1
            tl_parity_col_index = j - i%2                 
            tl_data_row_index = i - 1
            tl_data_col_index = 2*(j+1) - i%2 - 1

            
            ##### for edges and corners
            ######## if even row and col==max or if odd row and col==0
            ######## set data of max-col or 0-col of same row if it exists, else row - 1

            # Grab the values if they exist ---  should be combined across layers for ME
            # the "& ..." portion is for measurement errors - remove them if focus is only data errors

            # Index for center ancilla of the clique
            center_inx = (i*num_syndrome_cols) + j

            # Index for leaf ancillas of the clique
            tr_syn_inx = (tr_parity_row_index*num_syndrome_cols) + tr_parity_col_index
            br_syn_inx = (br_parity_row_index*num_syndrome_cols) + br_parity_col_index
            bl_syn_inx = (bl_parity_row_index*num_syndrome_cols) + bl_parity_col_index
            tl_syn_inx = (tl_parity_row_index*num_syndrome_cols) + tl_parity_col_index

            # Index for data qubits covered by the clique
            tr_data_inx = (tr_data_row_index*distance) + tr_data_col_index
            br_data_inx = (br_data_row_index*distance) + br_data_col_index
            bl_data_inx = (bl_data_row_index*distance) + bl_data_col_index
            tl_data_inx = (tl_data_row_index*distance) + tl_data_col_index


            center_value = (1-prev_syndrome[center_inx]) & (curr_syndrome[center_inx]) # this is the center
            if 0 <= tr_parity_row_index < num_syndrome_rows and 0 <= tr_parity_col_index < num_syndrome_cols:
                tr_value = (1-prev_syndrome[tr_syn_inx]) & (curr_syndrome[tr_syn_inx])
            else:
                tr_value = -1
            if 0 <= br_parity_row_index < num_syndrome_rows and 0 <= br_parity_col_index < num_syndrome_cols:
                br_value = (1-prev_syndrome[br_syn_inx]) & (curr_syndrome[br_syn_inx])
            else:
                br_value = -1
            if 0 <= bl_parity_row_index < num_syndrome_rows and 0 <= bl_parity_col_index < num_syndrome_cols:
                bl_value = (1-prev_syndrome[bl_syn_inx]) & (curr_syndrome[bl_syn_inx])
            else:
                bl_value = -1
            if 0 <= tl_parity_row_index < num_syndrome_rows and 0 <= tl_parity_col_index < num_syndrome_cols:
                tl_value = (1-prev_syndrome[tl_syn_inx]) & (curr_syndrome[tl_syn_inx])
            else:
                tl_value = -1

            #updating the input array --- TODO (added this as extra)
            #and_val = input_array[i][j] & input_array_old[i][j]
            #input_array[i][j] = input_array[i][j] ^ and_val # useful for ME next round
            ##### now check for the conditions from the paper #####

            count=0
            iscomplex=0
            if(center_value==1):
                if(tr_value==1):
                    count+=1
                if(br_value==1):
                    count+=1
                if(bl_value==1):
                    count+=1
                if(tl_value==1):
                    count+=1
                    
                if(count%2==0): 
                    #first check if this is an edge or corner
                    if((i%2==0 and j==(num_syndrome_cols-1)) or (i%2==1 and j==0)):
                        #this is an edge or corner
                        iscomplex=0
                        #setting one of two options for edges and a specific one for corner
                        if(i<num_syndrome_rows-1): 
                            row=i
                        else:
                            row=i-1
                        if(j==0):
                            col=0
                        else:
                            col=num_syndrome_cols-1
                        output_array[(row*distance) + col] = 1
                    else:
                        ### no this is not an edge or corner
                        iscomplex=1
                        return (output_array, iscomplex)
                else:
                    #do the corrections - this is set to one because correction means we are 
                    #forcing an X (or Z) on this data qubit.
                    if(tr_value==1):
                        output_array[tr_data_inx] = 1
                    if(br_value==1):
                        output_array[br_data_inx] = 1
                    if(bl_value==1):
                        output_array[bl_data_inx] = 1
                    if(tl_value==1):
                        output_array[tl_data_inx] = 1
                    
    return (output_array, iscomplex)

def baseline_clique(syndrome, distance, trial_num):
     #decoder decodes syndrome bits (this is a original but updated version of clique)
     #the input to this is 2 syndrome arrays 
    # the first is the current and the second is the old one.
    #this is done to combine two rounds of measurements as is discussed in the paper
     #this is only tackling one type of error (i.e. x or z type, which is sufficient0
    """
    Generate an n x n output array based on the input array.

    Parameters:
        2 input_arrays (list of lists): The input array of size (n+1) * ((n-1)/2).

    Returns:
        output_array (list of lists): The generated output array of size n x n.
    """
    output_array = np.zeros(distance*distance, dtype=np.uint8)
    
    num_syndrome_rows = distance+1
    num_syndrome_cols = (distance-1)//2

    for i in range(num_syndrome_rows):
        for j in range(num_syndrome_cols):
            # top right
            tr_parity_row_index = i - 1
            tr_parity_col_index = j + 1 - i%2
            tr_data_row_index = i - 1
            tr_data_col_index = 2*(j+1) - i%2   
            # bottom right
            br_parity_row_index = i + 1
            br_parity_col_index = j + 1 - i%2
            br_data_row_index = i
            br_data_col_index = 2*(j+1) - i%2
            # bottom left
            bl_parity_row_index = i + 1
            bl_parity_col_index = j - i%2   
            bl_data_row_index = i
            bl_data_col_index =  2*(j+1) - i%2 - 1
            # top left
            tl_parity_row_index = i - 1
            tl_parity_col_index = j - i%2                 
            tl_data_row_index = i - 1
            tl_data_col_index = 2*(j+1) - i%2 - 1

            # Index for center ancilla of the clique
            center_inx = (i*num_syndrome_cols) + j

            # Index for leaf ancillas of the clique
            tr_syn_inx = (tr_parity_row_index*num_syndrome_cols) + tr_parity_col_index
            br_syn_inx = (br_parity_row_index*num_syndrome_cols) + br_parity_col_index
            bl_syn_inx = (bl_parity_row_index*num_syndrome_cols) + bl_parity_col_index
            tl_syn_inx = (tl_parity_row_index*num_syndrome_cols) + tl_parity_col_index

            # Index for data qubits covered by the clique
            tr_data_inx = (tr_data_row_index*distance) + tr_data_col_index
            br_data_inx = (br_data_row_index*distance) + br_data_col_index
            bl_data_inx = (bl_data_row_index*distance) + bl_data_col_index
            tl_data_inx = (tl_data_row_index*distance) + tl_data_col_index

            center_value = syndrome[center_inx]

            tr_value = -1
            br_value = -1
            bl_value = -1
            tl_value = -1
            if 0 <= tr_parity_row_index < num_syndrome_rows and 0 <= tr_parity_col_index < num_syndrome_cols:
                tr_value = syndrome[tr_syn_inx]
            if 0 <= br_parity_row_index < num_syndrome_rows and 0 <= br_parity_col_index < num_syndrome_cols:
                br_value = syndrome[br_syn_inx]
            if 0 <= bl_parity_row_index < num_syndrome_rows and 0 <= bl_parity_col_index < num_syndrome_cols:
                bl_value = syndrome[bl_syn_inx]
            if 0 <= tl_parity_row_index < num_syndrome_rows and 0 <= tl_parity_col_index < num_syndrome_cols:
                tl_value = syndrome[tl_syn_inx]

            ##### now check for the conditions from the paper #####
            count=0
            iscomplex=0
            if(center_value==1):
                if(tr_value==1):
                    count+=1
                if(br_value==1):
                    count+=1
                if(bl_value==1):
                    count+=1
                if(tl_value==1):
                    count+=1
                    
                if(count%2==0):
                    iscomplex = 0
                    # Check top right corner
                    if (i == 0) and (j == (num_syndrome_cols - 1)):
                        # Correct top right data qubit
                        row = i
                        col = distance-1
                    
                    # Check bottom left corner
                    elif (i == (num_syndrome_rows - 1)) and (j == 0):
                        # Correct bottom left data qubit
                        row = i-1
                        col = 0

                    # Check if this is an edge
                    elif (i % 2 == 0 and j == (num_syndrome_cols - 1)) or (i % 2 == 1 and j == 0):
                        # Edge decoder throws complex when both leaf syndromes are active
                        if (count == 2):
                            iscomplex = 1
                        else:
                            row = i-1
                            if i % 2 == 0:
                                col = distance-1
                            else:
                                col = 0
                    else:
                        iscomplex = 1
                            
                    if iscomplex:
                        return (output_array, iscomplex)
                    output_array[(row*distance) + col] = 1

                else:
                    #do the corrections
                    if(tr_value==1):
                            output_array[tr_data_inx] = 1
                    if(br_value==1):
                            output_array[br_data_inx] = 1
                    if(bl_value==1):
                            output_array[bl_data_inx] = 1
                    if(tl_value==1):
                            output_array[tl_data_inx] = 1

    return (output_array,iscomplex)

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

def pinball_clique_late_meas(prev_syndrome, curr_syndrome, distance, trial_num):
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

def pinball_clique(syndrome, distance, trial_num):
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
                value1 = syndrome[center_inx] # this has to exist

                if 0 <= parity_row_index < num_syndrome_rows and 0 <= parity_col_index < num_syndrome_cols:
                    value2 = syndrome[neighbor_inx]
                else:
                    value2 = -1 #due to clockwise this is okay

                if(0 <= data_row_index < distance and 0 <= data_col_index < distance and value2 != -1):
                    andval = value1 & value2
                    output_array[data_inx] ^= andval
                    syndrome[center_inx] ^= andval
                    if 0 <= parity_row_index < num_syndrome_rows and 0 <= parity_col_index < num_syndrome_cols:
                        syndrome[neighbor_inx] ^= andval   
                        
                # in general we would want to do this stage later because it only consumes one syndrome for an error 
                # it doesnt really matter here because this never steals from the above i.e. a 2 is always a 2.
                if(0 <= data_row_index < distance and 0 <= data_col_index < distance and value2 == -1):
                    value2=1
                    andval = value1 & value2
                    output_array[data_inx] ^= andval
                    syndrome[center_inx] ^= andval
                    if 0 <= parity_row_index < num_syndrome_rows and 0 <= parity_col_index < num_syndrome_cols:
                        syndrome[neighbor_inx] ^= andval                           

                ## These last lines effectively make this a 4-stage pipeline in order to avoid combinational loops
            
    ## additional code for last col - this could be combined with als pipe stage
    for i in range(num_syndrome_rows): 
        if(i%2==1): # only for even rows
            continue
        j = num_syndrome_cols - 1

        center_inx = i*num_syndrome_cols + j

        value1 = syndrome[center_inx]
        if(value1):
            #BR should always work
            data_col_index = distance-1
            data_row_index = i
            output_array[(data_row_index*distance) + data_col_index] ^= 1
            syndrome[center_inx] ^= 1
    
    # If any syndromes have been left as "1", this indicates that a complex
    # decode is required
    iscomplex = np.any(syndrome)

    return (output_array, iscomplex)

def get_sim_inputs(distance, error_rate):
    data_errors = []
    syndromes = []
    
    data_in = f"sim/inputs/d_{distance}/e_{error_rate:.6f}/data_array.in"
    parity_in = f"sim/inputs/d_{distance}/e_{error_rate:.6f}/parity_array.in"
        
    # Read data errors in each trial
    with open(data_in, "r") as f:
        lines = f.readlines()
        for line in lines:
            # Flat array of data errors
            data_errors.append(np.array([int(c) for c in line.strip()], dtype=np.uint8))
    
    # Read syndromes in each trial
    with open(parity_in, "r") as f:
        lines = f.readlines()
        for line in lines:
            # Flat array of syndromes
            syndromes.append(np.array([int(c) for c in line.strip()], dtype=np.uint8))
        
    # Generate parity check matrix for surface code
    check = utils.generate_X_parity_check_matrix(d)

    return data_errors, syndromes, check

def python_sim(inp_N, inp_probability,inp_trials, decoder):

    d = inp_N  # Size of the array (number of rows/columns)
    e = inp_probability  # Probability of each element being 1 (50% chance)
    TOTAL = inp_trials #number of trials
    corrections = []

    num_syndrome_rows = d+1
    num_syndrome_cols = (d-1)//2

    legit_trials=0
    success_count=0
    complex_trials = []
    failed_trials = []
    
    data_errors, syndromes, check_matrix = get_sim_inputs(d, e)

    #now do the decoding on layer i, looking at measurement errors in conjunction with layer (i-1).
    for i in range(TOTAL):
        # syndromes for current round
        current_array = syndromes[i]

        # syndromes for next round
        if (i < TOTAL-1):
            next_array = syndromes[i+1]
        else:
            next_array = np.zeros(len(syndromes[i]), dtype=np.uint8)

        # Only pass true data error syndromes to the decoder if not in last round of batch
        if (i % d != d-1):
            post_meas_syndrome = current_array & (1-next_array)
        # Run the Python Clique decoder
        correction, iscomplex = decoder(post_meas_syndrome, d, i)

        # Handle clearing of second half of measurment error
        if (i % d != d-1):
            for j in range(num_syndrome_rows):
                for k in range(num_syndrome_cols):
                    inx = (j*num_syndrome_cols) + k
                    next_array[inx] = next_array[inx] ^ (current_array[inx] & next_array[inx])

        corrections.append(correction)
        if(iscomplex==1): #this means that the chain is complex and should be left for a full decoder
            complex_trials.append(i)
            continue

        legit_trials+=1 #this is the trials which are not deemed complex

        # Adds corrections (either these cancel out errors OR form loops OR form chains OR indicate unsuccessful decode)
        xor_result = np.bitwise_xor(data_errors[i], corrections[i]) 
        isfailure= utils.verify_decoder_corrections(xor_result, check_matrix) #check the corrections for logical failure
        
        if(isfailure==0):
            success_count+=1
        else:
            failed_trials.append(i)
    
    return (success_count, legit_trials)

def postprocess_verilog_sim(code_distance, error_rate, decoder_type, num_trials):    
    legit_trials=0 # counts number of non-complex trials
    success_count=0 # counts number of successful decodes performed 
    complex_trials = []
    failed_trials = []
    
    d = code_distance
    e = error_rate

    data_errors, syndromes, check_matrix = get_sim_inputs(d, e)

    if decoder_type == baseline_clique:
        sim_output = f"sim/outputs/verilog/baseline/d={code_distance}_e={error_rate:.6f}.out"
    else:
        sim_output = f"sim/outputs/verilog/pinball/d={code_distance}_e={error_rate:.6f}.out"
    
    corrections=[] # corrections applied
    complex_array=[] # complex detection events
    
    # Read simulation outputs
    with open(sim_output) as f:
        lines = f.readlines()
        for line in lines:
            arr = line.strip().split(",")

            # Read complex flags and corrections for each trial
            complex_array.append(int(arr[0]))
            corrections.append(np.array([int(c) for c in arr[1]], dtype=np.uint8))

    for i in range(num_trials):
        if complex_array[i]: # complex decode was required, skip
            complex_trials.append(i)
            continue

        legit_trials += 1

        # adds corrections (either these cancel out errors OR form loops OR form chains OR indicate unsuccessful decode)
        xor_result = np.bitwise_xor(data_errors[i], corrections[i]) 

        # Check if the decode performed for this trial was successful
        if not utils.verify_decoder_corrections(xor_result, check_matrix):
            success_count += 1
        else:
            failed_trials.append(i)
    
    return (success_count, legit_trials)

def parse_simulation_args():
    parser = argparse.ArgumentParser()

    # Parse command-line options
    parser.add_argument("-n", "--num_trials", help="Number of error trials to simulate")
    parser.add_argument("-e", "--error_rates", help="Physical error rates to simulate", nargs="*")
    parser.add_argument("-d", "--distances", help="Code distances to simulate", nargs="*")
    parser.add_argument("-t", "--decoder_type", help="Decoder type to simulate. Valid options are baseline, pinball.")
    parser.add_argument("-p", "--python", help="Run simulations with Python decoder", action="store_true")
    parser.add_argument("-v", "--verilog", help="Run simulations with Verilog decoder", action="store_true")
    parser.add_argument("-g", "--graph", help="Generate graphs of simulation outputs", action="store_true")

    args = parser.parse_args()

    num_trials = 100000
    if args.num_trials:
        num_trials = int(args.num_trials)
        if num_trials < 0:
            print("[ERROR] Negative number of trials specified.")
            exit(1)
    
    distances = [3,5,7,9]
    error_rates = [0.01, 0.005, 0.001, 0.0005, 0.0001]
    if args.distances:
        distances = [int(d) for d in args.distances]
        for d in distances:
            if not (d % 2):
                print("[ERROR] Only odd code distances can be simulated.")
                exit(1)
            elif d < 0:
                print("[ERROR] Negative code distance specified.")
                exit(1)
    if args.error_rates:
        error_rates = [float(e) for e in args.error_rates]

        for e in error_rates:
            if e < 0:
                print("[ERROR] Negative physical error rate specified.")
                exit(1)
    
    decoder_type = baseline_clique
    if args.decoder_type:
        if args.decoder_type == "baseline":
            decoder_type = baseline_clique
        elif args.decoder_type == "pinball":
            decoder_type = pinball_clique
        else:
            print(f"[ERROR] Invalid decoder type {args.decoder_type} specified. Valid options are baseline, pinball.")
            exit(1)

    # Process python simulation by default if nothing is specified
    # Otherwise, process whichever simulations are specified
    python = True
    verilog = False
    if args.verilog:
        verilog = True
        if not args.python:
            python = False
    
    graph = args.graph

    return (num_trials, error_rates, distances, decoder_type, python, verilog, graph)
    
if __name__ == "__main__":
    
    (num_trials, error_rates, distances, decoder_type, python, verilog, graph) = parse_simulation_args()
    color_series = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    if python:
        print("[INFO] Gathering Python decoder statistics...")

        success_counts_per_distance = []
        coverages_per_distance = []

        for d in distances:
            success_count_per_error_rate = []
            coverage_per_error_rate = []
            for p_err in error_rates:
                print(d,p_err)

                success_count, coverage = python_sim(d, p_err, num_trials, decoder_type)                
                success_count_per_error_rate.append(success_count)
                coverage_per_error_rate.append(coverage)

                print("done. Coverage % = ", coverage * 100 / num_trials)
                print("done. Decoding accuracy % = ", success_count * 100 / coverage)

                if decoder_type == baseline_clique:
                    with open(f"sim/outputs/stats/python/baseline/d={d}_e={p_err:.6f}.stats", "w") as s:
                        s.write(f"{coverage/num_trials}, {success_count/coverage}")
                else:
                    with open(f"sim/outputs/stats/python/pinball/d={d}_e={p_err:.6f}.stats", "w") as s:
                        s.write(f"{coverage/num_trials}, {success_count/coverage}")

            success_counts_per_distance.append(success_count_per_error_rate)
            coverages_per_distance.append(coverage_per_error_rate)
    
        if graph:
            legend = []
            for d in distances:
                legend.append(f"d={d}")
            
            plt.figure()
            utils.graph_coverage(distances, error_rates, num_trials, coverages_per_distance, color_series)
            plt.legend(legend)
            plt.savefig("coverage_python.png", dpi=100)
            
            plt.figure()
            utils.graph_decoding_accuracy(distances, error_rates, num_trials, coverages_per_distance, success_counts_per_distance, color_series)
            plt.legend(legend)
            plt.savefig("decoding_accuracy_python.png", dpi=100)

    if verilog:
        print("[INFO] Gathering Verilog decoder statistics...")

        success_counts_per_distance = []
        coverages_per_distance = []

        for d in distances:
            success_count_per_error_rate = []
            coverage_per_error_rate = []
            for p_err in error_rates:
                print(d,p_err)

                success_count, coverage = postprocess_verilog_sim(d, p_err, decoder_type, num_trials)                
                success_count_per_error_rate.append(success_count)
                coverage_per_error_rate.append(coverage)

                print("done. Coverage % = ", coverage * 100 / num_trials)
                print("done. Decoding accuracy % = ", success_count * 100 / coverage)

                if decoder_type == baseline_clique:
                    with open(f"sim/outputs/stats/verilog/baseline/d={d}_e={p_err:.6f}.stats", "w") as s:
                        s.write(f"{coverage/num_trials}, {success_count / coverage}")
                else:
                    with open(f"sim/outputs/stats/verilog/pinball/d={d}_e={p_err:.6f}.stats", "w") as s:
                        s.write(f"{coverage/num_trials}, {success_count / coverage}")
            
            success_counts_per_distance.append(success_count_per_error_rate)
            coverages_per_distance.append(coverage_per_error_rate)

        if graph:
            legend = []
            for d in distances:
                legend.append(f"d={d}")
            
            plt.figure()
            utils.graph_coverage(distances, error_rates, num_trials, coverages_per_distance, color_series)
            plt.legend(legend)
            plt.savefig("coverage_verilog.png", dpi=100)
            
            plt.figure()
            utils.graph_decoding_accuracy(distances, error_rates, num_trials, coverages_per_distance, success_counts_per_distance, color_series)
            plt.legend(legend)
            plt.savefig("decoding_accuracy_verilog.png", dpi=100)
