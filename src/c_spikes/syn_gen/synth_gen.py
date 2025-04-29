'''
%%%
% Routine for creation of synthetic datasets from GCaMP biophysical model
% 
% Inputs:
%   -spike_rate - the mean spike rate of the dataset to create synthetic data
%      from
%   -spike_params - list of 2 scalars
%      - time constant for the smoothing of the varying rate in seconds [default 5 seconds]
%      - ratio between 0 and 1 of the time with non-zero rate [default 0.5]
%   -cell_params -  the cell paramters for the biophysical model extracted
%      by the MC approach for Bayesian inference
%   -noise_dir - path to directory with ground truth noise samples
%   -GCaMP_model - pulling from the pgas library for this - better approach?
%   -tag - dataset-specific tag
%
%
%%%
'''

import numpy as np
import scipy.io as sio
import os
from scipy.stats import norm


class synth_gen():
  def __init__(self, spike_rate=2, spike_params=[5, 0.5],cell_params=[30e-6, 10e2, 1e-5, 5, 30, 10],
    noise_dir="gt_noise_dir", GCaMP_model=None, tag="default", plot_on=False,use_noise=True,noise_val=2):
    
    # Get current directory of this file, prepend to noise_dir
    self.noise_dir = noise_dir
    full_noise_path = self.noise_dir

    # Synth data settings
    self.spike_rate = spike_rate
    self.spike_params = spike_params
    self.Cparams = cell_params

    
    # Determine noise directory and get the file list, get path elements
    self.noise_files = [os.path.join(full_noise_path, f) for f in os.listdir(full_noise_path) if f.endswith('.mat')]
    self.tag = tag
    #Handling noise cases
    self.use_noise = use_noise
    self.noise_val = noise_val
    
    # Load the GCaMP_model
    self.gcamp = GCaMP_model
    
    #For QC plots
    self.plot_on = plot_on

  def calculate_standardized_noise(self,dff,frame_rate):
    noise_levels = np.nanmedian(np.abs(np.diff(dff, axis=-1)), axis=-1) / np.sqrt(frame_rate)
    return noise_levels * 100     # scale noise levels to percent

    
  def spk_gen(self,T):
    """
    Generate a simulated spike train.
    """
    spike_rate = self.spike_rate
    spike_params = self.spike_params
    
    # Set generator parameters
    #print(f"spike_params[0] = {spike_params[0]}")
    smoothtime = spike_params[0]
    rnonzero = spike_params[1]
    
    # 1) generate the varying rate vector
    # get gaussian
    nsub = 20
    dt = smoothtime / nsub
    T = np.ceil(T)
    #print(f"T = {T} and dt = {dt}")
    nt = int(np.ceil(T / dt))
    x = np.random.randn(nt + 2 * nsub)
    tau = np.array([nsub, 0])
    dim = 0
    s1 = x.shape
    xf = np.fft.fft(x, axis=dim)
    nk = s1[dim]
    freqs = np.fft.fftfreq(nk)
    freq2 = freqs ** 2
    HWHH = np.sqrt(2 * np.log(2))
    freqthr = 1. / tau[0]
    sigma = freqthr / HWHH
    K = 1. / (2 * sigma ** 2)
    K = np.array([K])
    K[np.isinf(K)] = 1e6
    g = np.exp(-K[0] * freq2)
    # apply gaussian in fourrier space
    xf = xf * g
    y = np.fft.ifft(xf, axis=dim)
    vrate = np.real(y)
    vrate = vrate[nsub:(nsub + nt)]
    # calculate std of elem of vrate
    s = nsub * HWHH / (2 * np.pi)
    sr = np.sqrt(1 / (2 * np.sqrt(np.pi) * s))
    # and rescale to 1
    vrate = vrate / sr
    # translate, and scale the rate vector
    thr = norm.ppf(1 - rnonzero)
    vrate = np.maximum(0, vrate - thr)
    vrate = self.spike_rate*vrate/np.max((vrate))
    
    #2 generate spikes
    # choose sampling rate with at most one spike per bin
    dt1 = 1 / np.max(vrate) / 100
    nt1 = int(np.floor(T / dt1))
    t = np.linspace(0, T, nt1)
    vrate_interp = np.interp(t, np.arange(nt) * dt, vrate)
    # make spikes
    nspike = (np.random.rand(nt1).reshape(-1,1) < vrate_interp * dt1)
    
    spikes = np.nonzero(nspike)[0] * dt1
    spikes = self.strip_sub_refrac(spikes)
    
    if self.plot_on:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(t, vrate_interp)
        for spike in spikes:
            plt.axvline(x=spike, color='k', linestyle='--')
        plt.show()
    
    return spikes
  
  def strip_sub_refrac(self, spikes):
    """
      A utility to strip out any spikes that come at intervals below a refractory period of 1 ms.
    """
    ISIs = np.diff(spikes)
    sub_refrac_idx = np.where(ISIs < 0.001)[0] + 1
    spikes = np.delete(spikes, sub_refrac_idx)
    
    return spikes
    
  def set_synth_params(self,spike_rate=None,spike_params=None,cell_params=None):
    """
      Method to allow resetting of the synthetic generator parameters
    """
    if spike_rate:
      self.spike_rate = spike_rate
    if spike_params:
      self.spike_params = spike_params
    if cell_parmas:
      self.cell_params = cell_params
  
  def generate(self,Cparams=None,tag=None,output_folder="."):
    """
      This generates the actual synthetic data. Need to add some randomization stuff
    """
    #Make dir to hold the synthetic data
    self.tag = tag if tag else self.tag
    synth_dir = os.path.join(output_folder, "Ground_truth", f"synth_{self.tag}")
    #print(synth_dir)
    os.makedirs(synth_dir,exist_ok=True)
    
    #Pass in Cparams and initialize the model
    Cparams = Cparams if Cparams else self.Cparams
    self.gcamp.setParams(Cparams[0],Cparams[1],Cparams[2],Cparams[3],\
      Cparams[4],Cparams[5])
    self.gcamp.init()
    
    # This is for if we'd like to plot one output
    test_plot = True
    
    # Loop over and add synthetic simulations to each noise file
    for file in self.noise_files:
      # load CASCADE-formatted noise traces
      noise_data = sio.loadmat(file)
      CAttached = noise_data['CAttached']
      keys = CAttached[0][0].dtype.descr
      
      for ii in np.arange(len(CAttached[0])):
        existing_fields = CAttached[0][ii].dtype.names
        new_fields = []
        if 'fluo_mean' not in existing_fields:
            new_fields.append(('fluo_mean', '|O'))
        if 'events_AP' not in existing_fields:
            new_fields.append(('events_AP', '|O'))

        # Add new field for noise + simulation
        new_inner_dtype = np.dtype(CAttached[0][ii].dtype.descr + \
          new_fields)
        
        #Set up structure to hold fluo_mean
        inner_array = CAttached[0][ii]
        new_inner_array = np.empty(inner_array.shape, dtype=new_inner_dtype)
        
        # Copy existing data to the new structured array
        for field in inner_array.dtype.names:
            new_inner_array[field] = inner_array[field]
            
        #Noise, time, get spikes
        noise = inner_array['gt_noise'][0][0]
        time = inner_array['fluo_time'][0][0]
        T = time[-1] - time[0]

        #Check standardized noise and toss if over criterion
        if self.use_noise:
          frame_rate = 1/np.mean(np.diff(time.flatten()))
          standard_noise = self.calculate_standardized_noise(noise.T,frame_rate.T)
          if standard_noise>self.noise_val:
            break

        try:
          
          spikes = self.spk_gen(T) + time[0]
          
          #Simulation
          self.gcamp.integrateOverTime(time.flatten().astype(float), spikes.flatten().astype(float))
          dff_clean = self.gcamp.getDFFValues()
          
          #Add new field and fill with sim + noise and spike times
          #new_inner_array['fluo_mean'] = np.empty((1, 1), dtype='|O')
          new_inner_array['fluo_mean'][0][0] = np.array([dff_clean.flatten() + noise.flatten()])
          #new_inner_array['events_AP'] = np.empty((1, 1), dtype='|O')
          
          new_inner_array['events_AP'][0][0] = spikes.reshape(-1,1)*1e4
        except Exception:
          break
        
        CAttached[0][ii] = new_inner_array
        #print("CAttached[0][ii].dtype.descr = ",CAttached[0][ii].dtype.descr)
      
      basename = os.path.basename(file)
      file_name, extension = os.path.splitext(basename)
      fname = 'rate='+str(self.spike_rate) + 'param='+str(self.spike_params[0])+ '_'+ str(self.spike_params[1]) +\
        file_name + extension
      save_path = os.path.join(synth_dir, fname)
      print("syn_gen save_path = ", save_path)
      sio.savemat(save_path, {'CAttached': CAttached})
      
      # reset the model (might want to add this to the end of the c++ method to reduce exposure
      self.gcamp.init()
      
      if self.plot_on and test_plot:
        test_plot = False
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(time.flatten(), dff_clean.flatten())
        #plt.plot(time.flatten(), noise.flatten())
        for spike in spikes:
          plt.axvline(x=spike,color='k',linestyle='-')
        plt.show()
        

if __name__ == "__main__":
  synth = synth_gen(plot_on=False)
  #synth.spk_gen(np.array([200]))
  synth.generate()
