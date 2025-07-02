import numpy as np
from scipy.linalg import null_space 
import json
import os
from scipy.interpolate import interp1d
import math

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
    def __init__(self,params=None,spikes=None,duration=1,dt=1e-9,dt_out=1e-4, use_linear_approx=False):
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
        self.use_linear_approx = use_linear_approx

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
        if 'fixed_buffer' in self.params:
            self.kon_B  = self.params['fixed_buffer']['kon_B']
            self.koff_B = self.params['fixed_buffer']['koff_B']
            self.B_tot  = self.params['fixed_buffer']['B_tot']
        else:
            self.kon_B = None
        ## ATP
        if 'ATP' in self.params:
            self.kon_ATP = self.params['ATP']['kon_ATP']
            self.koff_ATP = self.params['ATP']['koff_ATP']
            self.ATP_tot = self.params['ATP']['ATP_tot']
        else:
            self.kon_ATP = None
        ## Free buffer - trun off if not included
        if 'free_buffer' in self.params:
            self.kon_free = self.params['free_buffer']['kon_free']
            self.koff_free = self.params['free_buffer']['koff_free']
            self.free_tot = self.params['free_buffer']['free_tot']
        else:
            self.kon_free = None
        # Calretinin
        if 'calretinin' in self.params:
            cr = self.params['calretinin']
            self.CR_tot = cr['CR_tot']  # Total calretinin concentration (M)
            self.k1on_CR = cr['k1on']
            self.k1off_CR = cr['k1off']
            self.k2on_CR = cr['k2on']
            self.k2off_CR = cr['k2off']
            self.kon_CR = True
            self.TR0 = 0.0  # will be updated in initialCondition
            self.RCaRCa0 = 0.0
        else:
            self.kon_CR = None
        # Parvalbumin
        if 'parvalbumin' in self.params:
            pv = self.params['parvalbumin']
            self.PV_tot = pv['PV_tot']
            self.k1on_PV = pv['k1on']    # Mg binding (1/(M·s))
            self.k1off_PV = pv['k1off']
            self.k2on_PV = pv['k2on']   # Ca binding (1/(M·s))
            self.k2off_PV = pv['k2off']
            self.kon_PV = True
            self.PB0 = 0.0
            self.PBCa0 = 0.0
            self.Mg0 = 1e-3
        else:
            self.kon_PV = None

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
        
        n_states = 11
        # Get initial system state and edge states
        if self.kon_B is not None:
            n_states += 1
            self.BCa0 = self.kon_B*self.c0/(self.kon_B*self.c0 +self.koff_B)*self.B_tot

            Kd_B = self.koff_B / self.kon_B
            self.kap_B = self.B_tot / Kd_B

        if self.kon_ATP is not None:
            n_states += 1
            self.ATP0 = self.kon_ATP*self.c0/(self.kon_ATP*self.c0 +self.koff_ATP)*self.ATP_tot

            Kd_ATP = self.koff_ATP / self.kon_ATP
            self.kap_ATP = self.ATP_tot / Kd_ATP

        if self.kon_free is not None:
            n_states += 1
            self.free0 = self.kon_free*self.c0/(self.kon_free*self.c0 +self.koff_free)*self.free_tot

            Kd_free = self.koff_free / self.kon_free
            self.kap_free = self.free_tot / Kd_free

        if self.kon_CR is not None:
            n_states += 2
            K1 = self.k1on_CR / self.k1off_CR
            K2 = self.k2on_CR / self.k2off_CR
            ca = self.c0

            denom = 1 + K1*ca + K1*K2*ca**2
            TR0 = K1 * ca * self.CR_tot / denom
            RCaRCa0 = K1 * K2 * ca**2 * self.CR_tot / denom

            self.TR0 = TR0
            self.RCaRCa0 = RCaRCa0

            Kd_CR1 = self.k1off_CR / self.k1on_CR
            Kd_CR2 = self.k2off_CR / self.k2on_CR
            #ChatGPT Suggestion
            self.kap_CR = self.CR_tot * (1 / Kd_CR1 + 2 / Kd_CR2)

            #From Faas et. al 2007
            #self.kap_CR = self.CR_tot / math.sqrt(Kd_CR1 * Kd_CR2)

        if self.kon_PV is not None:
            n_states += 2
            K1 = self.k1on_PV / self.k1off_PV
            K2 = self.k2on_PV / self.k2off_PV
            ca = self.c0
            mg = self.Mg0  # Assume a constant Mg2+ concentration

            r1 = mg * K2 / (ca * K1)
            r2 = ((r1 * self.k1off_PV + self.k2off_PV) / (mg * self.k1on_PV + ca * self.k2on_PV)) + r1 + 1
            PB0 = self.PV_tot / r2
            PBCa0 = PB0 * self.k2on_PV * ca / self.k2off_PV

            self.PB0 = PB0
            self.PBCa0 = PBCa0

            # PV kinetic constants
            k1on = self.k1on_PV  # Mg2+ on
            k1off = self.k1off_PV
            k2on = self.k2on_PV  # Ca2+ on
            k2off = self.k2off_PV

            # Effective Ca2+ binding fraction accounting for Mg2+ occupancy
            # From Lee 2000 (Chromaffin...), Equation 7:
            numerator = self.PV_tot * k2on / k2off
            denominator = 1 + (mg * k1on / k1off)
            self.kap_PV = numerator / denominator

        self.n_states = n_states

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

        initialCondition = np.concatenate((G,[self.c0,self.c0]))

        if self.kon_B is not None:
            initialCondition = np.concatenate((initialCondition,[self.BCa0]))
        if self.kon_ATP is not None:
            initialCondition = np.concatenate((initialCondition,[self.ATP0]))
        if self.kon_free is not None:
            initialCondition = np.concatenate((initialCondition,[self.free0]))
        if self.kon_CR is not None:
            initialCondition = np.concatenate((initialCondition, [self.TR0, self.RCaRCa0]))
        if self.kon_PV is not None:
            initialCondition = np.concatenate((initialCondition, [self.PB0, self.PBCa0]))

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
    
    def model(self, state, calcium_input):
        if self.use_linear_approx:
            return self.model_la(state, calcium_input)
        else:
            return self.model_full(state, calcium_input)
        

    def model_full(self,state,calcium_input):
        
        dt = self.dt

        # Separate states for matmul
        G = state[:9]
        Ca = state[9]
        Ca_in = state[10]

        state_ind_shift = 0

        # calcium flux due to GCaMP
        Gflux = self.getFlux(Ca,G)

        # Taylor expansion at current time
        dG_dt = np.matmul(self.Gmat(Ca),G)

        if self.kon_B is not None:
            BCa = state[11]
            dBCa_dt = self.kon_B*(self.B_tot-BCa)*Ca - self.koff_B*BCa
            BCa = BCa +  dt*dBCa_dt
            state[11] = BCa
            state_ind_shift += 1
        else:
            dBCa_dt = 0

        if self.kon_ATP is not None:
            ATP = state[11 + state_ind_shift]
            dATP_dt = self.kon_ATP*(self.ATP_tot-ATP)*Ca - self.koff_ATP*ATP
            ATP = ATP + dt*dATP_dt
            state[11 + state_ind_shift] = ATP
            state_ind_shift += 1
        else:
            dATP_dt = 0
        
        # Handle free separately
        if self.kon_free is not None:
            free = state[11 + state_ind_shift]
            dfree_dt = self.kon_free*(self.free_tot-free)*Ca - self.koff_free*free
            free = free + dt*dfree_dt
            state[11 + state_ind_shift] = free
        else:
            dfree_dt = 0

        # Calretinin
        if self.kon_CR is not None:
            TRCa = state[11 + state_ind_shift]
            RCaRCa = state[12 + state_ind_shift]
            TT = self.CR_tot - TRCa - RCaRCa

            dTRCa_dt = self.k1on_CR * TT * Ca - self.k1off_CR * TRCa \
                    - self.k2on_CR * TRCa * Ca + self.k2off_CR * RCaRCa
            dRCaRCa_dt = self.k2on_CR * TRCa * Ca - self.k2off_CR * RCaRCa

            # Update states
            TRCa += dt * dTRCa_dt
            RCaRCa += dt * dRCaRCa_dt
            state[11 + state_ind_shift] = TRCa
            state[12 + state_ind_shift] = RCaRCa

            dCR_dt = dTRCa_dt + 2 * dRCaRCa_dt
        else:
            dCR_dt = 0

        # Parvalbumin
        if self.kon_PV is not None:
            PB = state[11 + state_ind_shift]
            PBCa = state[12 + state_ind_shift]
            PBMg = self.PV_tot - PB - PBCa

            dPB_dt = PBMg * self.k1off_PV - PB * self.Mg0 * self.k1on_PV \
                    - PB * Ca * self.k2on_PV + PBCa * self.k2off_PV
            dPBMg_dt = PB * self.Mg0 * self.k1on_PV - PBMg * self.k1off_PV # Unused since not modeling Mg2+ flux
            dPBCa_dt = PB * Ca * self.k2on_PV - PBCa * self.k2off_PV

            PB += dt * dPB_dt
            PBCa += dt * dPBCa_dt

            state[11 + state_ind_shift] = PB
            state[12 + state_ind_shift] = PBCa

            dPV_dt = dPBCa_dt
        else:
            dPV_dt = 0

        dCa_dt  = -self.gamma*(Ca-self.c0)\
            -self.gam_in*(Ca - self.c0) + self.gam_out*(Ca_in - self.c0) \
            + Gflux - dBCa_dt + calcium_input - dATP_dt - dfree_dt - dCR_dt - dPV_dt
        dCa_in_dt = self.gam_in*(Ca - self.c0) - self.gam_out*(Ca_in - self.c0)
        
        # State updates
        G   = G   +  dt*dG_dt
        Ca  = Ca  +  dt*dCa_dt
        Ca_in = Ca_in + dt*dCa_in_dt

        # Pack out
        state[:9] = G
        state[9] = Ca
        state[10] = Ca_in

        return(state)
    
    def model_la(self, state, calcium_input):
        """
        Linear Approximation (LA) model using simplified buffer capacity approximation.
        Includes Mg²⁺ competition for parvalbumin (PV) as per Neher (1992).
        """
        dt = self.dt

        # Extract current state variables
        G = state[:9]
        Ca = state[9]
        Ca_in = state[10]

        # --- GCaMP dynamics ---
        Gflux = self.getFlux(Ca, G)
        dG_dt = np.matmul(self.Gmat(Ca), G)

        dBCa_dt = 0
        dATP_dt = 0
        dfree_dt = 0
        dCR_dt = 0
        dPV_dt = 0

        # Fixed buffer
        if self.kon_B is not None:
            dBCa_dt = (self.kap_B / (1 + self.kap_B)) * Ca

        # ATP
        if self.kon_ATP is not None:
            dATP_dt = (self.kap_ATP / (1 + self.kap_ATP)) * Ca

        # Free buffer
        if self.kon_free is not None:
            dfree_dt = (self.kap_free / (1 + self.kap_free)) * Ca

        # Calretinin (2-site; simplified additive model)
        if self.kon_CR is not None:
            dCR_dt = (self.kap_CR / (1 + self.kap_CR)) * Ca

        # Parvalbumin with Mg²⁺ competition
        if self.kon_PV is not None:
            dPV_dt = (self.kap_PV/ (1 + self.kap_PV)) * Ca

        buffer_sink = dBCa_dt + dATP_dt + dfree_dt + dCR_dt + dPV_dt

        # Calcium dynamics with buffer-adjusted influx term
        dCa_dt = -self.gamma * (Ca - self.c0) \
                - self.gam_in * (Ca - self.c0) + self.gam_out * (Ca_in - self.c0) \
                + Gflux + calcium_input - buffer_sink  

        dCa_in_dt = self.gam_in * (Ca - self.c0) - self.gam_out * (Ca_in - self.c0)

        # --- Update states ---
        G += dt * dG_dt
        Ca += dt * dCa_dt
        Ca_in += dt * dCa_in_dt

        # --- Pack updated state ---
        state[:9] = G
        state[9] = Ca
        state[10] = Ca_in

        return state
      

    def euler(self,n_sim=None):
        # With n_sim we specify the number of time points we would like in the sim
        if n_sim is None:
            n_sim = int(np.ceil(self.duration / self.dt)) + 1

        # Check if we're using free, adjust state number accordingly
        n_states = self.n_states #if not self.use_linear_approx else 11
        print(n_states)

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
            if (i % 100000) == 0:
                print(f"iteration: {i}")
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
    duration = 0.01
    t_out = np.arange(0, duration, dt_out)
    #spikes = np.linspace(0, 0.05, 1)
    #spikes = np.array([0.01])
    rate_hz = 10
    isi = np.random.exponential(1.0 / rate_hz, size=int(rate_hz * duration * 2))  # oversample
    spike_times = np.cumsum(isi)
    spikes = spike_times[spike_times < duration]
    spikes = [0]

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
    pFile = r'parameter_files/params_fe_cal.json'
    spike_time = 0
    # I spoofed an overloaded constructor here with conditionals (check out top lines of __init__ to see what I mean)
    # This will cause initializations to be calculated, etc.
    gcamp_dt_out = 1e-4
    gcamp = GCaMP(params=pFile,spikes=spikes,duration=duration,
                  dt_out=gcamp_dt_out, use_linear_approx=True)
    print(gcamp.spike_times)
    # Properly speaking, I should have segregated private and public methods here, but suffice to say this one is public
    # input is number of iterations - default is to use duration/dt steps in not passed in
    iter = int(duration / gcamp.dt)
    start = time.time()
    states,dff = gcamp.euler(iter)
    end = time.time()
    print(f"gcamp.euler() took {end - start:.6f} seconds")
    #time_np = np.arange(0, duration + gcamp_dt_out, gcamp_dt_out)
    time_np = np.arange(0, duration, gcamp_dt_out)
    #states,dff = gcamp.euler()
    # QC plot comparing 1e-5 and finer time basis (could go finer, but memory lags after a while)
    plt.figure(figsize=(8, 4))
    plt.plot(time_np, dff[:-1])
    plt.title("DFF")
    plt.show()
    plt.figure(figsize=(8, 4))
    plt.plot(states[:,-2])
    plt.title("States")
    plt.show()

    tag = "test_cal_la_model_upd_kap"
    np_sav_dir = os.path.join("results/sim_output", tag)
    
    if not os.path.exists(np_sav_dir):
        os.makedirs(np_sav_dir)
    else:
        os.makedirs(np_sav_dir + "_x")

    np.save(os.path.join(np_sav_dir, "dff.npy"), dff)
    np.save(os.path.join(np_sav_dir, "states.npy"), states)
    np.save(os.path.join(np_sav_dir, "time.npy"), time_np)
    np.save(os.path.join(np_sav_dir, "spike_times.npy"), spikes)
    