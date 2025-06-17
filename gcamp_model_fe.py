import numpy as np
from scipy.linalg import null_space 
import json
from scipy.interpolate import interp1d

def gauss_sum_interp(
    spike_times, duration, dt_out=None, sigma=1.18e-4, offset=4e-4, area_per_spike=1.0, dt_kernel=1e-5
):
    """
    Converts spike times to calcium fluxes. Allows optional full
    interpolation onto time basis of dt_out as QC.

    """
    n_kernel = int(np.ceil((offset + 5 * sigma) / dt_kernel)) + 1
    t_kernel = np.arange(0, n_kernel) * dt_kernel
    kernel = np.exp(-((t_kernel - offset) ** 2) / (2 * sigma ** 2))
    kernel *= area_per_spike / (kernel.sum() * dt_kernel)
    n_trace = int(np.ceil(duration / dt_kernel)) + 1
    t_trace = np.arange(n_trace) * dt_kernel
    trace = np.zeros(n_trace, dtype=float)

    spike_indices = (np.asarray(spike_times) / dt_kernel).round().astype(int)
    valid = spike_indices < n_trace
    spike_indices = spike_indices[valid]

    for idx in spike_indices:
        end = min(n_trace, idx + len(kernel))
        trace[idx:end] += kernel[:end - idx]

    if dt_out is not None:
        # Interpolate onto t_out
        n_out = int(np.ceil(duration / dt_out)) + 1
        t_out = np.arange(n_out) * dt_out
        interp_func = interp1d(t_trace, trace, kind='linear', bounds_error=False, fill_value=0)
        ca_interp = interp_func(t_out)

        # Area preservation: scale so that sum*dt matches original
        area_orig = trace.sum() * dt_kernel
        area_interp = ca_interp.sum() * np.mean(np.diff(t_out))
        if area_interp > 0:
            ca_interp *= (area_orig / area_interp)
    else:
        t_out = None
        ca_interp = None

    return t_trace, trace, t_out, ca_interp

class GCaMP:
    """
    A class to simulate GCaMP calcium indicator dynamics based on provided parameters and spike times.
    """
    def __init__(self,params=None,spikes=None,duration=1,dt=1e-9,dt_out=1e-4):
        """
        Initialize the GCaMP model with parameters and spike times.
        :param params: Path to the JSON file containing model parameters.
        :param spikefile: Path to the text file containing spike times.
        """
        # Set simulation time basis sufficiently small to ensure numerical stability
        # but interpolate out on something smaller
        self.dt = dt
        self.dt_out = dt_out
        self.duration = duration

        # Load the parameters and spike times
        if params is None:
            paramsfile = r'params.json'
            self.params = json.load(open(paramsfile))
        elif isinstance(params, str):
            paramfile = params
            self.params = json.load(open(paramfile))
        else:
            self.params = params

        if spikes is None:
            spikefile = r'spike_times.txt'
            self.spike_times = np.loadtxt(spikefile)
        elif isinstance(spikes, str):
            spikefile = spikes
            self.spike_times = np.loadtxt(spikefile)
        else:
            self.spike_times = spikes
        
        #Load parameters from JSON file
        Gparams = self.params['Gparams']
        # Explicitly state Gparams, calculating where needed
        self.konN = Gparams[1]/(Gparams[0]**Gparams[2]);self.koffN=Gparams[1]
        self.konC = Gparams[4]/(Gparams[3]**Gparams[5]);self.koffC=Gparams[4]
        self.H_N = Gparams[2]; self.H_C = Gparams[5]
        
        self.konPN = Gparams[6]; self.koffPN = Gparams[7]
        self.konPC = Gparams[8]; self.koffPC = Gparams[9]

        self.konPN2 = Gparams[10]; self.koffPN2 = Gparams[11]
        self.konPC2 = Gparams[12]; self.koffPC2 = Gparams[13]

        # Load the Cparams
        self.G_tot = self.params['Cparams']['G_tot']
        self.gamma = self.params['Cparams']['gamma']
        self.DCaT = self.params['Cparams']['DCaT']
        self.Rf = self.params['Cparams']['Rf']
        self.gam_in = self.params['Cparams']['gam_in']
        self.gam_out = self.params['Cparams']['gam_out']

        # Load the buffer parameters
        ## Fixed buffer
        self.kon_B  = self.params['fixed_buffer']['kon_B']
        self.koff_B = self.params['fixed_buffer']['koff_B']
        self.B_tot  = self.params['fixed_buffer']['B_tot']
        ## ATP
        self.kon_ATP = self.params['ATP']['kon_ATP']
        self.koff_ATP = self.params['ATP']['koff_ATP']
        self.ATP_tot = self.params['ATP']['ATP_tot']
        ## Free buffer - trun off if not included
        if 'free_buffer' in self.params:
            self.kon_free = self.params['free_buffer']['kon_free']
            self.koff_free = self.params['free_buffer']['koff_free']
            self.free_tot = self.params['free_buffer']['free_tot']
        else:
            self.kon_free = None

        # Constants you probably won't change, but just to explicitly state them
        self.FWHM = 2.8e-4 # FWHM for calcium flux (s)
        self.Ca_sat = 1e-2 # saturating calcium for GCaMP (M)
        self.c0 = 5e-8 # resting calcium concentration (M)
        self.bright_states = np.array([2,3,5,6,8])

        # Get the spike-dependent calcium inputs (returned as kernels with specified
        # temporal offset on time basis dictated by dt_kernel = so we don't need a 
        # huge, uniformative vector on the simulation time basis)
        self.t_kernel,self.ca_influx_kernel,_,_ = gauss_sum_interp(self.spike_times, self.duration, dt_out=None,
                sigma=self.FWHM/2.3548, offset=4e-4,
                area_per_spike=self.DCaT,dt_kernel=1e-5)

        # Get initial system state and edge states
        self.BCa0 = self.kon_B*self.c0/(self.kon_B*self.c0 +self.koff_B)*self.B_tot
        self.ATP0 = self.kon_ATP*self.c0/(self.kon_ATP*self.c0 +self.koff_ATP)*self.ATP_tot
        if self.kon_free is not None:
            self.free0 = self.kon_free*self.c0/(self.kon_free*self.c0 +self.koff_free)*self.free_tot
        # GCaMP initializations
        self.Ginit = self.getInitialCondition(self.c0)[self.bright_states].sum()
        self.G0 = self.getInitialCondition(0)[self.bright_states].sum()
        self.Gsat = self.getInitialCondition(self.Ca_sat)[self.bright_states].sum()


    def Gmat(self,ca):
        konN   = self.konN*ca**self.H_N;koffN  = self.koffN
        konC   = self.konC*ca**self.H_C;koffC  = self.koffC

        konPN  = self.konPN;koffPN = self.koffPN
        konPC  = self.konPC;koffPC = self.koffPC
        konPN2   = self.konPN2;koffPN2  = self.koffPN2
        konPC2   = self.konPC2;koffPC2  = self.koffPC2

        Gmat = np.array([[-(konN+konC),koffN,koffPN,0,koffC,koffPC,0,0,0],
		    [konN,-(koffN+konPN+konC),0,0,0,0,koffPC2,koffC,0],
			[0,konPN,-(konC+koffPN),koffC,0,0,0,0,koffPC2],
			[0,0,konC,-(koffC+konPC2+koffPN2 ),0,0,0,konPN2,0],
			[konC,0,0,koffPN2,-(koffC+konPC+konN),0,0,koffN,0],
			[0,0,0,0,konPC,-(koffPC+konN),koffN,0,koffPN2],
			[0,0,0,0,0,konN,-(koffN+konPN2+koffPC2),konPC2,0],
			[0,konC,0,0,konN,0,0,-(koffC+koffN+konPC2+konPN2),0],
			[0,0,0,konPC2,0,0,konPN2,0,-(koffPN2+koffPC2)]])

        return(Gmat)
    
    def getInitialCondition(self,ca):
        # Null space for solving the steady state system
        ns = null_space(self.Gmat(ca))[:,0]
        G=ns/sum(ns)*self.G_tot
        # Loads into initial state as G, BCa, Ca, Ca_in, ATP, free
        if self.kon_free is not None:
            initialCondition = np.concatenate((G,[self.BCa0,self.c0,self.c0,self.ATP0,self.kon_free]))
        else:
            initialCondition = np.concatenate((G,[self.BCa0,self.c0,self.c0,self.ATP0]))

        return(initialCondition)

    def getFlux(self,ca,G):
        konN   = self.konN*ca**self.H_N
        koffN  = self.koffN
        konC   = self.konC*ca**self.H_C
        koffC  = self.koffC

        koffPN = self.koffPN
        koffPC = self.koffPC
        koffPN2  = self.koffPN2
        koffPC2  = self.koffPC2

        #Calculate fluxes per binding site
        N = -(konN*(G[0] + G[4] + G[5])) +\
                koffN*(G[1] + G[6]+ G[7])  + \
                koffPN*G[2]  + \
                koffPN2*(G[3] + G[8])
        
        C = -(konC*(G[0] + G[1] + G[2])) +\
                koffC*(G[4] + G[3] + G[7])  + \
                koffPC*G[5] + \
                koffPC2*(G[6] + G[8])

        Csites=2
        return(2*N + Csites*C)

    def model(self,state,calcium_input):
        
        dt = self.dt

        # Separate states for matmul
        G = state[:9]
        BCa = state[9]
        Ca = state[10]
        Ca_in = state[11]
        ATP = state[12]

        # calcium flux due to GCaMP
        Gflux = self.getFlux(Ca,G)

        # Taylor expansion at current time
        dG_dt = np.matmul(self.Gmat(Ca),G)
        dBCa_dt = self.kon_B*(self.B_tot-BCa)*Ca - self.koff_B*BCa
        dCa_dt  = -self.gamma*(Ca-self.c0)\
            -self.gam_in*(Ca - self.c0) + self.gam_out*(Ca_in - self.c0) \
            + Gflux - dBCa_dt + calcium_input
        dCa_in_dt = self.gam_in*(Ca - self.c0) - self.gam_out*(Ca_in - self.c0)
        dATP_dt = self.kon_ATP*(self.ATP_tot-ATP)*Ca - self.koff_ATP*ATP
        
        # Handle free separately
        if self.kon_free is not None:
            free = state[13]
            dfree_dt = self.kon_free*(self.free_tot-free)*Ca - self.koff_free*free
            free = free + dt*dfree_dt
            state[13] = free
        
        # State updates
        G   = G   +  dt*dG_dt
        BCa = BCa +  dt*dBCa_dt
        Ca  = Ca  +  dt*dCa_dt
        Ca_in = Ca_in + dt*dCa_in_dt
        ATP = ATP + dt*dATP_dt

        # Pack out
        state[:9] = G
        state[9] = BCa
        state[10] = Ca
        state[11] = Ca_in
        state[12] = ATP
        return(state)

    def euler(self,n_sim=None):
        # With n_sim we specify the number of time points we would like in the sim
        if n_sim is None:
            n_sim = int(np.ceil(self.duration / self.dt)) + 1

        # Check if we're using free, adjust state number accordingly
        n_states = 14 if self.kon_free is not None else 13

        # Set up
        dt=self.dt
        state_out=np.zeros((n_sim,n_states))
        initial_state = self.getInitialCondition(self.c0)
        state_out[0,:]=initial_state
        # this is for interpolating the coarse gaussian kernels onto the simulation time basis
        interp_func = interp1d(self.t_kernel, self.ca_influx_kernel, kind='linear', bounds_error=False, fill_value=0)
        T = 0 # current simulation time (s)

        # forward euler
        for i in range(1,n_sim):
            # Get the calcium input
            calcium_input = interp_func(T)
            state_out[i,:] = self.model(state_out[i-1,:],calcium_input)
            T+=dt
        
        # Interpolate onto a reasonable time basis
        t_sim = np.arange(n_sim) * self.dt
        n_out = int(np.ceil(self.duration / self.dt_out)) + 1
        t_out = np.arange(n_out) * self.dt_out
        interp_func_out = interp1d(t_sim, state_out, axis=0, kind='linear', bounds_error=False, fill_value=0)
        state_interp = interp_func_out(t_out)

        # Calculate dff from the interpolated state values
        Gbright = state_interp[:, self.bright_states].sum(axis=1)
        dff = (Gbright - self.Ginit)/\
            (self.Ginit-self.G0+(self.Gsat-self.G0)/(self.Rf-1))

        return state_interp, dff
            
if __name__ == '__main__':
    # Demo code for a few key functions
    import matplotlib.pyplot as plt
    import time

    ### Demonstration for the input calcium function
    # params
    dt_kernel = 1e-5
    dt_out = 1e-6
    DCaT = 1e-5
    duration = 1
    t_out = np.arange(0, duration, dt_out)
    spikes = np.linspace(0, 0.1, 10)

    # Calc over dt=1e-5 time basis, time it
    start = time.time()
    t_kernel, trace_kernel, t, ca = gauss_sum_interp(spikes, duration, dt_out=dt_out,
                sigma=1.18e-4, offset=4e-4,
                area_per_spike=DCaT,dt_kernel=dt_kernel)
    end = time.time()
    print(f"gauss_sum_interp took {end - start:.6f} seconds")

    # QC plot comparing 1e-5 and finer time basis (could go finer, but memory lags after a while)
    plt.figure(figsize=(8, 4))
    plt.plot(t, ca, label="Interpolated Calcium Trace")
    plt.plot(t_kernel,trace_kernel,label="Kernel basis",linestyle="--")
    plt.vlines(spikes, ymin=0, ymax=np.max(ca), color='r', linestyle='--', label="Spike Time")
    plt.text(0.05, 0.95, f"sum ca {ca.sum()*dt_out:.4e}", transform=plt.gca().transAxes)
    plt.xlabel("Time (s)")
    plt.ylabel("Calcium Entry (a.u.)")
    plt.title("Area-Preserving Interpolated Calcium Trace")
    plt.legend()
    plt.tight_layout()
    plt.show()

    ### Deomnstration for the GCaMP class
    pFile = r'params.json'
    spike_time = 0
    # I spoofed an overloaded constructor here with conditionals (check out top lines of __init__ to see what I mean)
    # This will cause initializations to be calculated, etc.
    gcamp = GCaMP(params=pFile,spikes=spike_time,duration=1)
    # Properly speaking, I should have segregated private and public methods here, but suffice to say this one is public
    # input is number of iterations - default is to use duration/dt steps in not passed in
    states,dff = gcamp.euler(2000000)
    # QC plot comparing 1e-5 and finer time basis (could go finer, but memory lags after a while)
    plt.figure(figsize=(8, 4))
    plt.plot(dff)
    plt.show()
    plt.figure(figsize=(8, 4))
    plt.plot(states[:,-2])
    plt.show()
    