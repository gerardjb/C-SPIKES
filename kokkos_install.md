# Instalation
Vcpkg is used to grab the c++ dependecies. However, I was running into issues with later vcpkg compatability with the kokkos pin. Current solution is to use an older vcpkg tag as below:
```bash
cd /scratch/gpfs/<username>/C-SPIKES
git clone https://github.com/microsoft/vcpkg
./vcpkg/bootstrap-vcpkg.sh
cd vcpkg
git checkout tags/2025.04.09
cd ..
./vcpkg/vcpkg install gsl Armadillo jsoncpp boost-circular-buffer
./vcpkg/vcpkg integrate install
```
And then our HPC dropin modules for cuda so:
```bash
module load cudatoolkit/12.9
```
then best to make an editable local install unless you don't plan to make changes to the codebase:
```bash
pip install -e .
```
You can add the -v flag to the pip call for the scikit build core if you'd like more verbosity on the error messages, etc., during the build.
