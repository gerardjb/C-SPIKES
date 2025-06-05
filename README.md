# High level remarks
Ah, I'll start by saying, C-SPIKES stands for (Calcium Spike Processing using Integrated Kinetic Estimation and Simulation).
Which I personally feel is awesome.
In the setup as of this push, I've made three branches: main, model_eval_WIP and Sanjeev_colab.
Start on Sanjeev_colab. There, your mission is as we discussed in [this doc](https://docs.google.com/document/d/18tyZVOgEhgmnUVLgcIlCuda2tce7VNL6WAcFc87xzmc/edit?usp=sharing).
Basically determine the robustness of the SMC and ML approaches if your Cparams are 
very different than the cell on which you're trying to do inference.

# How to (hopefully) super-easy set up on your linux-based HPC platform

The plan goes in two parts: set up vcpkg (although I'll note that Dave and I are trying
to minimize how much of that we'll need in the final wheel - not relevant to you right now)
followed by the actual installation of the project. So, let's do that step-by-step.

## Setup vcpkg
All of this stuff is verbatim (minus repo name changes and the particulars of your HPC paths) from Dave's original recommendations
as we haven't been able to wean ourselves from the c++ modules that require this junk.
One day, maybe. Meantime, you have to do this to set up the package installers for the c++
piece of the project that will be important for you.
```bash
cd /scratch/gpfs/<username>/TEST2
git clone https://github.com/microsoft/vcpkg
./vcpkg/bootstrap-vcpkg.sh
./vcpkg/vcpkg install gsl Armadillo jsoncpp boost-circular-buffer
./vcpkg/vcpkg integrate install
```

## Setting up your conda-type environment
I know nothing whatsoever about the drop-in conda modules that are on Anschultz's HPCs
expect that they almost certainly exist. So, you'll need to do the standard module loadings
as you were accustomed to on della. But once that's done, the project is now set up to where
you can create the environment, clone the repo, and install easily as below (note that 3.10
seems like a good middle ground for now, but that may change):
```bash
conda create -n c_spikes python=3.10
conda activate c_spikes
pip install -e .
```
Here, it's important to remind that you can add optional modules to install along with the
core scikit build project with syntax like (see the toml file if you're curious about other
stuff dave put into the original options):
```bash
pip install -e .[jupyter]
```
if you want to install things like jupyter notebook-relevant modules in the conda environment.
But scikit build handles all of the c++ library linking (including tensorflow and pytorch 
and all the stupid Armadilo crap!) in the core build along with specifying how imports should
be arranged via path specification whether you're doing an an editable install (like us since
we inclde the "-e" flag) or making a proper wheel. It's like a kind of magic.

## Environment variable stuff
(What follows is not strictly necessary, just shows how to control environment variable state
on conda environment activation. But if you're curious, proceed...)
Once your environment is set up, you can take advantage of the pragma compiler directives that
Dave added to the code back in the hackathon days by setting the OpenMP thread number as:
```bash
export OMP_NUM_THREADS=<x_num_threads>
```
As I aluded to at our meeting, you can also have a standing order for number of threads that you
want to call that gets set up on your conda environment activation. Basically, you can edit your
"~/.conda/envs/c_spikes/etc/conda/de<>activate.d" files to capture the environment variable state
when you activate the c_spikes environment. Could do that something like this in your bash scripts
in the .d directories with activation like:
```bash
# Remember the old OMP values paths for when we deactivate
export OLD_OMP_NUM_THREADS=$OMP_NUM_THREADS
export OMP_NUM_THREADS=<x_num_threads>
```
and then for the deactivation include:
```bash
# Reset the old OMP values paths for when we deactivate
export OMP_NUM_THREADS=$OLD_OMP_NUM_THREADS
```
which is really just to be overly-cautious that you're not changing a previous .bashrc configuration
for something else you may be doing.
