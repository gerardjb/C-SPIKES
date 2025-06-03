#%%
#Importing project packages and required libraries
import numpy as np
import c_spikes.pgas.pgas_bound as pgas
from c_spikes.syn_gen import synth_gen
import matplotlib.pyplot as plt
import scipy.io as sio
import os


#Setting flags for what to calculate on this run
recalc_pgas = False # this runs the particle Gibbs sampler to extract spike times and cell parameters
recalc_Cparams = False # this runs the particle Gibbs sampler with known spike times to extract cell parameters
recalc_synth = False # this runs the synthetic data generation code to create new synthetic data
retrain_and_infer = True # this runs the cascade training and inference code to train a new model and infer spikes on the original data
model_source = "ens2"   # FLAG: choose "cascade" or "ens2"

#Utility-type methods
def spike_times_2_binary(spike_times,time_stamps):
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

# For opening the janelia datasets
def open_Janelia_1(j_path):
    all_data = sio.loadmat(j_path)
    dff = all_data['dff']
    time_stamps = all_data['time_stamps']
    spike_times = all_data['ap_times'] 

    return time_stamps, dff, spike_times

# For calculating noise levels in the data
def calculate_standardized_noise(dff,frame_rate):
    noise_levels = np.nanmedian(np.abs(np.diff(dff, axis=-1)), axis=-1) / np.sqrt(frame_rate)
    return noise_levels * 100     # scale noise levels to percent

"""
    Setup for the demo
"""

# First we'll load in the original data as a numpy array
#janelia_file = "jGCaMP8f_ANM471993_cell03" (high SNR excitatory)#"jGCaMP8f_ANM478349_cell06" (low SNR inhibitory)
janelia_file = "jGCaMP8f_ANM471993_cell03"
filename = os.path.join("gt_data",janelia_file+".mat")

time,data,spike_times = open_Janelia_1(filename)
time1 = np.float64(time[0,1000:2000])
time1 = time1.copy()
data1 = np.float64(data[0,1000:2000])
data1 = data1.copy()
binary_spikes = np.float64(spike_times_2_binary(spike_times,time1))
#binary_spikes = np.float64([0])

# Run the particle Gibbs sampler to extract cell parameters
## Setting up parameters for the particle gibbs sampler
tag="test_param_out"
Gparam_file="src/c_spikes/pgas/20230525_gold.dat"

# Set file name for Cparams
output_folder = os.path.join("results","pgas_output")
os.makedirs(output_folder,exist_ok=True)
param_sample_file = os.path.join(output_folder,"param_samples_"+tag+".dat")

"""
    Here will be the PGAS run for spike extraction and demo use (plotting, etc.)
"""
if recalc_pgas:
    # Set up the parameters for the particle gibbs sampler
    analyzer = pgas.Analyzer(
        time=time1,
        data=data1,
        constants_file="parameter_files/constants_GCaMP8_soma.json",
        output_folder="results/pgas_output",
        column=1,
        tag=tag,
        niter=2,
        append=False,
        verbose=1,
        gtSpikes=binary_spikes,
        has_gtspikes=True,
        maxlen=1000, 
        Gparam_file=Gparam_file,
        seed=2
    )

    ## Run the sampler
    analyzer.run()

    # <xxx pick up here - next need to port MAP procedures and select best spike train (in logp_xxx_test_param_out.dat)
    # then need to run the sampler again with the MAP spike train and get Cparams
    # then link to the synthetic data generation code and cascade training/inference xxx>

    ## Return and print the output
    #parameter_samples = analyzer.get_parameter_estimates()

    # maybe some plots of the spikes here. Have to think a bit about what to include
    # could be a saved file showing spike predictions, DFF, and gt spikes

"""
    Run of the particle Gibbs sampler to extract cell parameters
"""

if recalc_Cparams:
    analyzer = pgas.Analyzer(
        time=time1,
        data=data1,
        constants_file="parameter_files/constants_GCaMP8_soma.json",
        output_folder="results/pgas_output",
        column=1,
        tag=tag,
        niter=300,
        append=False,
        verbose=1,
        gtSpikes=binary_spikes,
        has_gtspikes=True,
        maxlen=1000, 
        Gparam_file=Gparam_file,
        seed=2
    )

    ## Run the sampler
    analyzer.run()

    ## Return and print the output
    parameter_samples = analyzer.get_parameter_samples()   
else:
    ## Opening the saved parameter samples for use as Cparams
    parameter_samples = np.loadtxt(param_sample_file,delimiter=',',skiprows=1)

"""
    Syntetic data generation procedures
"""

# Prepare Cparams calculation from parameter estimates - less than 100 samples for testing
burnin = 100 if np.size(parameter_samples,0) > 100 else 0
parameter_samples = parameter_samples[burnin:,0:6]
print("mean of samples")
print(np.mean(parameter_samples,axis=0))


# Only recompute the synthetic data if necessary
if recalc_synth:
    # Create synthetic data
    ## Load parameters into the GCaMP model to use for synthetic data creation
    Cparams = np.mean(parameter_samples,axis=0)
    Gparams = np.loadtxt(Gparam_file)
    gcamp = pgas.GCaMP(Gparams,Cparams)

    ## Generate synthetic data from the PGAS-derived cell paramters 
    # Now making broader spike pulls
    nominal_rates = np.array([2])#
    for rate in nominal_rates:
        synth = synth_gen.synth_gen(plot_on=False,GCaMP_model=gcamp,\
            spike_rate=rate,cell_params=Cparams,tag=tag,use_noise=True, noise_dir="gt_noise_dir")
        synth.generate(output_folder=f"results")

"""
    Cascade training and inference procedures
"""

if retrain_and_infer:
    # First get the data and noise level we're training against
    time_stamps = time[0, :]
    fluo_data = data[0, :]
    frame_rate = 1/np.mean(np.diff(data[0,:]))
    from c_spikes.cascade2p import utils
    noise_level = utils.calculate_noise_levels(fluo_data,frame_rate)

    # Set configurations file for sample training
    synthetic_training_dataset = os.path.join(f"synth_{tag}")
    cfg = dict( 
        model_name = f"synth_trained_model{tag}",    # Model name (and name of the save folder)
        sampling_rate = 30,    # Sampling rate in Hz (round to next integer)
        training_datasets = [synthetic_training_dataset],
        noise_levels = [noise for noise in range(2,3)],#[int(np.ceil(noise_level)+1)]
        smoothing = 0.05,     # std of Gaussian smoothing in time (sec)
        causal_kernel = 0,   # causal ground truth smoothing kernel
        verbose = 1,
            )

    if model_source.lower() == "cascade":
        print("Using Cascade for training and inference")
        ## Package checks for cascade
        print("Current directory: {}".format(os.getcwd()))
        from c_spikes.cascade2p import checks, cascade
        print("\nChecks for packages:")
        checks.check_packages()

        # save parameter as config.yaml file - TODO: make cascade overwrite configs on this call
        print(cfg['noise_levels'])
        cascade.create_model_folder(cfg, model_folder="results/Pretrained_models")

        # Train a model based on config contents
        model_name = cfg['model_name']
        cascade.train_model( model_name, model_folder="results/Pretrained_models",\
            ground_truth_folder="results/Ground_truth")

        # Use trained model to perform inference on the original dataset
        from c_spikes.cascade2p.utils_discrete_spikes import infer_discrete_spikes
        spike_prob = cascade.predict(model_name, np.reshape(fluo_data, (1, len(fluo_data))), model_folder='results/Pretrained_modles')

        '''
        For now turning this part off -> convergence takes a long time if model predictions are low SNR
        which is a high probability during new model testing.
        '''
        #spike_prob = cascade.predict(model_name, fluo_data) # for multidimensional_data
        #discrete_approximation, spike_time_estimates = infer_discrete_spikes(spike_prob,model_name)

        # Separate Python file that organizes model names 
        ## Saving routine
        save_dir = "results/cascade_output/predictions"
        os.makedirs(save_dir,exist_ok=True)
        save_path = os.path.join(save_dir,f"{tag}_output.mat")
        sio.savemat(save_path,{'spike_prob':spike_prob,'time_stamps':time_stamps,'dff':fluo_data,'spike_times':spike_times,'cfg':cfg})#'spike_time_estimates':spike_time_estimates, <- this is for the discrete spike time estimates
    elif model_source.lower() == "ens2":
        ## ENS2 training and inference
        print("Using ENS2 model for training and inference")
        from c_spikes.ens2 import ens
        # Prepare data (same fluo_data and time_stamps as above)
        
        # Use the same sampling_rate and smoothing as CASCADE config for consistency
        sampling_rate = cfg.get('sampling_rate', 60)
        smoothing_std = cfg.get('smoothing', 0.050)
        # Train ENS2 model on synthetic data
        ens.train_model(tag, sampling_rate=sampling_rate, smoothing_std=smoothing_std, 
                         use_causal_kernel=False, verbose=2,batch_size=256)
        # Perform inference on the original trace
        spike_prob = ens.predict(tag, np.reshape(fluo_data, (1, len(fluo_data))), 
                                   sampling_rate=sampling_rate, smoothing_std=smoothing_std)
        # Save results (continuous spike probability output)
        save_dir = "results/ens2_output/predictions"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{tag}_ens2_output.mat")
        sio.savemat(save_path, {'spike_prob': spike_prob, 'time_stamps': time_stamps, 'dff': fluo_data,'spike_times':spike_times})
    else:
        raise ValueError("model_source must be 'cascade' or 'ens2'")