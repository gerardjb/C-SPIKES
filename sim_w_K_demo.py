# %%

import numpy as np
import os
import c_spikes.pgas.pgas_bound as pgas
import matplotlib.pyplot as plt


# Create test Cparams (G_tot, gamma, DCaT, Rf, gam_in, gam_out, koff_B, kon_B, B_tot)
Cparams = np.array([1e-5, 1000, 1e-5, 5, 40, 4, 1e4, 1e8, 0.14])
Cparam_file = "results/test_output/test_Cparams.txt"
# This is required for the GCaMP model to run, as it expects a ascii compliant file
os.makedirs(os.path.dirname(Cparam_file), exist_ok=True)  # Ensure the directory exists
np.savetxt(Cparam_file, Cparams, fmt="%.8e")  # arma::raw_ascii is just plain text

# Make a GCaMP model simulation
Gparam_file="src/c_spikes/pgas/20230525_gold.dat"
gcamp = pgas.GCaMP(Gparam_file, Cparam_file)
#Time/stims for the simulation
sampling_rate = 1000
sim_start = .0
sim_end = 0.15
sim_time = np.linspace(sim_start,sim_end,num=int(np.round((sim_end-sim_start)*sampling_rate)))
spike_times = np.array([0.01, 0.05, 0.7])


# Run the simulation and retrieve results
gcamp.integrateOverTime(sim_time.astype(float), spike_times.astype(float))
sim_dff = gcamp.getDFFValues()
# And for fun put on noise (like you'll need, Sanjeev)
noise_lev = 0.001
noise = np.random.normal(0,noise_lev, size=sim_dff.shape)
sim_dff_noisy = sim_dff + noise

# Plot the results
plt.figure()
plt.plot(sim_time, sim_dff_noisy, label='Simulated DFF with noise')
plt.plot(sim_time, sim_dff, label='Simulated DFF', linestyle='--')
plt.vlines(spike_times, ymin=np.min(sim_dff_noisy), ymax=np.max(sim_dff_noisy), color='red', alpha=0.7, label='Spikes')
plt.xlabel('Time (s)')
plt.ylabel('DFF')
plt.title('Simulated GCaMP DFF with Spikes')
plt.legend()
plt.xlim(sim_start, sim_end)

#plt.show()
plt.savefig("results/test_output/simulated_dff.png")
# %%
