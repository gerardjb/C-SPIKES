# General utility functions for the c_spikes package
import scipy.io as sio
import numpy as np

def load_Janlia_data(j_path):
    """
    Load Janelia dataset from a .mat file. This is for the data as in the custom
    imports done for Broussard et al. 2025 of Marton Rosza's Janelia dandi dataset
    released with the Zhang et al. 2023 paper describing the jGCaMP8 sensors.
    Args:
        j_path (str): Path to the .mat file containing Janelia dataset.
    Returns:
        tuple: Contains time_stamps, dff (fluorescence data), and spike_times.
    """

    all_data = sio.loadmat(j_path)
    dff = all_data['dff']
    time_stamps = all_data['time_stamps']
    spike_times = all_data['ap_times'] 

    return time_stamps, dff, spike_times

def spike_times_2_binary(spike_times,time_stamps):
    """
    Convert spike times to a binary vector based on provided time stamps.
    Args:
        spike_times (np.ndarray): Array of spike event times.
        time_stamps (np.ndarray): Array of time stamps to create binary vector against.
    Returns:
        np.ndarray: Binary vector where each element corresponds to a time stamp.
    """
    # initialize the binary vector
    binary_vector = np.zeros(len(time_stamps), dtype=int)

    # get event times within the time_stamps ends
    good_spike_times = spike_times[(spike_times >= time_stamps[0]) & (spike_times <= time_stamps[-1])]
    
    # Find the nearest element in 'a' that is less than the elements in 'b'
    for event_time in good_spike_times:
        # Find indices where 'a' is less than 'event_time'
        valid_indices = np.where(time_stamps < event_time)[0]
        if valid_indices.size > 0:
            nearest_index = valid_indices[-1]  # Taking the last valid index
            binary_vector[nearest_index] += 1

    return binary_vector