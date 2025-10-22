# Instalation
The vcpkg stuff is still the same as I haven't removed any library dependencies (yet). However, I was running into issues with later vcpkg compatability with the kokkos pin. Current solution is to use an older vcpkg tag as below:
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
And then I ended up using the cudatoolkit modules on della rather than a local kokkos install outside of the toml, so:
```bash
module load cudatoolkit/12.9
```
then I usually just do the basic editable install:
```bash
pip install -e .
```
where lately I've found the 'v' flag to be unneccesary (minus when I was getting the kokkos environment set up in the first place).
