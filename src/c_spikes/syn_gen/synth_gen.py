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

import os
import hashlib
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import scipy.io as sio
from scipy.stats import norm


class synth_gen():
  def __init__(self, spike_rate=2, spike_params=[5, 0.5],cell_params=[30e-6, 10e2, 1e-5, 5, 30, 10],
    noise_dir="gt_noise_dir", GCaMP_model=None, tag="default", plot_on=False,use_noise=False,noise_val=2,
    noise_fraction: float = 1.0, noise_seed: Optional[Union[int, Sequence[int]]] = None,
    log_nonfinite: bool = False):
    
    # Get current directory of this file, prepend to noise_dir
    base_dir = Path(__file__).resolve().parent
    noise_dir_path = Path(noise_dir)
    if not noise_dir_path.is_absolute():
      noise_dir_path = base_dir / noise_dir_path
    self.noise_dir = str(noise_dir_path)
    full_noise_path = self.noise_dir

    # Synth data settings
    self.spike_rate = spike_rate
    self.spike_params = spike_params
    self.Cparams = cell_params

    
    # Determine noise directory and store the file list
    self._all_noise_files = sorted([os.path.join(full_noise_path, f) for f in os.listdir(full_noise_path) if f.endswith('.mat')])
    self.noise_fraction = float(noise_fraction)
    # Normalize seeds to a list
    if noise_seed is None:
      self.noise_seeds = [None]
    elif isinstance(noise_seed, (list, tuple)):
      self.noise_seeds = list(noise_seed)
    else:
      self.noise_seeds = [noise_seed]
    self.tag = tag
    #Handling noise cases
    self.use_noise = use_noise
    self.noise_val = noise_val
    
    # Load the GCaMP_model
    if GCaMP_model is None:
      import c_spikes.pgas.pgas_bound as pgas

      # Default GCaMP parameter file colocated with the PGAS module
      gparam_path = Path(__file__).resolve().parents[1] / "pgas" / "20230525_gold.dat"
      Gparams = np.loadtxt(gparam_path)
      self.gcamp = pgas.GCaMP(Gparams, cell_params)
    else:
      self.gcamp = GCaMP_model
    
    #For QC plots
    self.plot_on = plot_on
    self.log_nonfinite = bool(log_nonfinite)

  def _seed_to_int(self, seed: Optional[int]) -> int:
    """Convert an optional seed to a stable uint32 value."""
    if seed is None:
      key = f"{self.noise_dir}|{float(self.noise_fraction):.6f}".encode("utf-8")
      seed_val = int(hashlib.sha256(key).hexdigest()[:8], 16)
    else:
      seed_val = int(seed) & 0xFFFFFFFF
    return seed_val

  def calculate_standardized_noise(self,dff,frame_rate):
    noise_levels = np.nanmedian(np.abs(np.diff(dff, axis=-1)), axis=-1) / np.sqrt(frame_rate)
    return noise_levels * 100     # scale noise levels to percent

    
  def spk_gen(self,T):
    """
    Generate a simulated spike train. Based loosely on pocedures described by Deneux et al., 2016.
    
    TODO: need more flexibilty around bursting here
    """
    spike_rate = self.spike_rate
    spike_params = self.spike_params
    
    # Set generator parameters
    smoothtime = spike_params[0]
    rnonzero = spike_params[1]
    
    # generate the varying rate vector
    # get gaussian
    nsub = 20
    dt = smoothtime / nsub
    T = np.ceil(T)
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
    # If sr<=0, return a single spike at dt*3
    if sr<=0:
      spikes = dt*3
      if self.log_nonfinite:
        print("Warning: NaN in vrate, returning single spike at dt*3")
      return spikes
    
    vrate = vrate / sr
    # translate, and scale the rate vector
    thr = norm.ppf(1 - rnonzero)
    vrate = np.maximum(0, vrate - thr)
    max_vrate = np.max(vrate)
    if not np.isfinite(max_vrate) or max_vrate <= 0:
      return np.array([], dtype=float)
    vrate = self.spike_rate * vrate / max_vrate
    

    #2 generate spikes
    # choose sampling rate with at most one spike per bin
    max_vrate = np.max(vrate)
    if not np.isfinite(max_vrate) or max_vrate <= 0:
      return np.array([], dtype=float)
    dt1 = 1 / max_vrate / 100
    if not np.isfinite(dt1) or dt1 <= 0:
      return np.array([], dtype=float)
    nt1 = int(np.floor(T / dt1)) if np.isfinite(T) else 0
    if nt1 <= 0:
      return np.array([], dtype=float)
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
    if spike_rate is not None:
      self.spike_rate = spike_rate
    if spike_params is not None:
      self.spike_params = spike_params
    if cell_params is not None:
      self.Cparams = cell_params
  
  def _select_noise_subset(self, seed: Optional[int]):
    """Deterministically select a subset of noise files based on fraction and seed."""
    if self.noise_fraction >= 1.0:
      return self._all_noise_files
    n_total = len(self._all_noise_files)
    n_keep = max(1, int(np.ceil(self.noise_fraction * n_total)))
    seed_val = self._seed_to_int(seed)
    rng = np.random.RandomState(seed_val)
    idx = rng.choice(n_total, size=n_keep, replace=False)
    return [self._all_noise_files[i] for i in sorted(idx)]


  def generate(self,Cparams=None,tag=None,output_folder="."):
    """
      This generates the actual synthetic data.
       TODO: add some randomization stuff
    """
    #Make dir to hold the synthetic data
    self.tag = tag if tag else self.tag
    synth_dir = os.path.join(output_folder, "Ground_truth", f"synth_{self.tag}")
    #print(synth_dir)
    os.makedirs(synth_dir,exist_ok=True)
    print(f"[syn_gen] Generating synthetic data to {synth_dir}")
    
    #Pass in Cparams and initialize the model
    Cparams = Cparams if Cparams else self.Cparams
    self.gcamp.setParams(Cparams[0],Cparams[1],Cparams[2],Cparams[3],\
      Cparams[4],Cparams[5])
    self.gcamp.init()
    
    # This is for if we'd like to plot one output
    test_plot = True
    
    # Loop over seeds and add synthetic simulations to each selected noise file
    for seed in self.noise_seeds:
      # Make spike generation reproducible for a given (noise_seed, noise_fraction).
      # This keeps sweep runs stable when using deterministic seed lists.
      np.random.seed(self._seed_to_int(seed))
      noise_files = self._select_noise_subset(seed)
      seed_suffix = f"_seed{seed}" if seed is not None else "_seedauto"
      for file in noise_files:
        # load CASCADE-formatted noise traces
        noise_data = sio.loadmat(file)
        CAttached = noise_data['CAttached']
        keys = CAttached[0][0].dtype.descr
        noisy_idxs = []
        
        for ii in np.arange(len(CAttached[0])):
          existing_fields = CAttached[0][ii].dtype.names
          new_fields = []
          if 'fluo_mean' not in existing_fields:
              new_fields.append(('fluo_mean', '|O'))
          if 'events_AP' not in existing_fields:
              new_fields.append(('events_AP', '|O'))
          if 'fluo_time' not in existing_fields:
              new_fields.append(('fluo_time', '|O'))

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

          # Hack for now - I didn't capture the time stamps in the noise files when I generated them
          try:
            time = inner_array['fluo_time'][0][0]
          except:
            sampling_rate = 121.9  # Hz
            n_samples = noise.shape[0]
            time = np.arange(n_samples) / sampling_rate
          T = time[-1] - time[0]
          
          # Generate spikes
          spikes = self.spk_gen(T) + time[0]
            
          # Handle cases where we have no spikes - spike at time point 3
          #if not len(spikes)>0: 
          #  spikes = time[2]
          
          # Simulation
          self.gcamp.integrateOverTime(time.flatten().astype(float), spikes.flatten().astype(float))
          dff_clean = self.gcamp.getDFFValues()

          # Clean non-finite values (replace with finite fallbacks) and log summary
          mask_dff = ~np.isfinite(dff_clean)
          mask_noise = ~np.isfinite(noise)
          mask_time = ~np.isfinite(time)
          mask_spikes = ~np.isfinite(spikes)
          if mask_dff.any() or mask_noise.any() or mask_time.any() or mask_spikes.any():
            if self.log_nonfinite:
              pct_dff = 100.0 * np.count_nonzero(mask_dff) / float(dff_clean.size if dff_clean.size else 1)
              pct_noise = 100.0 * np.count_nonzero(mask_noise) / float(noise.size if noise.size else 1)
              pct_time = 100.0 * np.count_nonzero(mask_time) / float(time.size if time.size else 1)
              pct_spk = 100.0 * np.count_nonzero(mask_spikes) / float(spikes.size if spikes.size else 1)
              msg = (
                f"[syn_gen] Cleaning non-finite values "
                f"dff={pct_dff:.3f}% noise={pct_noise:.3f}% time={pct_time:.3f}% spikes={pct_spk:.3f}%"
              )
              try:
                msg += f" file={os.path.basename(file)}"
              except Exception:
                pass
              try:
                msg += f" time_range=({np.nanmin(time):.3f},{np.nanmax(time):.3f})"
              except Exception:
                pass
              if spikes.size:
                try:
                  msg += f" spikes_range=({np.nanmin(spikes):.3f},{np.nanmax(spikes):.3f})"
                except Exception:
                  pass
              print(msg)
            # Replace NaN/inf with zeros for dff/noise, clip time/spikes by interpolation or fill
            if mask_dff.any():
              dff_clean = np.where(mask_dff, 0.0, dff_clean)
            if mask_noise.any():
              noise = np.where(mask_noise, 0.0, noise)
            if mask_time.any():
              # simple forward-fill/back-fill for time
              valid = np.isfinite(time)
              if valid.any():
                time = np.interp(np.arange(time.size), np.flatnonzero(valid), time[valid])
              else:
                time = np.linspace(0, T, time.size)
            if mask_spikes.any():
              spikes = spikes[np.isfinite(spikes)]
              if spikes.size == 0:
                spikes = np.array([], dtype=float)
          
          # Add new field and fill with sim + noise, spike times, and time
          new_inner_array['fluo_mean'][0][0] = np.array([2*dff_clean.flatten() + noise.flatten()])
          new_inner_array['events_AP'][0][0] = spikes.reshape(-1,1)*1e4
          new_inner_array['fluo_time'][0][0] = time.flatten()
          
          # Update the CAttached structure with the new inner array entries
          CAttached[0][ii] = new_inner_array
        
        # Remove noisy traces
        if len(noisy_idxs)>0:
          CAttached = np.delete(CAttached, noisy_idxs, axis=1)
        # Save the modified CAttached structure back to the file
        basename = os.path.basename(file)
        file_name, extension = os.path.splitext(basename)
        fname = 'rate='+str(self.spike_rate) + 'param='+str(self.spike_params[0])+ '_'+ str(self.spike_params[1]) +\
          seed_suffix + '_' + file_name + extension
        save_path = os.path.join(synth_dir, fname)
        sio.savemat(save_path, {'CAttached': CAttached})
        
        # reset the model (might want to add this to the end of the c++ method to reduce pybind exposure)
        self.gcamp.init()
        
    print(f"[syn_gen] Synthetic data generated at {synth_dir}")
        

if __name__ == "__main__":
  # For troubleshooting
  # Make a gcamp object
  import numpy as np
  import c_spikes.pgas.pgas_bound as pgas
  Gparam_file="src/c_spikes/pgas/20230525_gold.dat"
  Gparams = np.loadtxt(Gparam_file)
  Cparams=np.array([30e-6, 10e2, 1e-5, 5, 30, 10])
  gcamp = pgas.GCaMP(Gparams,Cparams)

  # Use gcamp model instantiation to create a synthetic generator
  rate = 2; tag = "test_synth"
  synth = synth_gen(plot_on=False,GCaMP_model=gcamp,\
    spike_rate=rate,cell_params=Cparams,tag=tag,use_noise=True, noise_dir="gt_noise_dir")
  synth.generate(output_folder=f"results")
