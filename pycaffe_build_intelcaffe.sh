#!/bin/bash -l
module load craype-haswell
module load PrgEnv-intel
module swap intel  intel/17.0.0.098
#module load cray-memkind
source /opt/intel/compilers_and_libraries/linux/mpi/intel64/bin/mpivars.sh intel64
module load curl/7.48.0
module load netcdf/4.4.1
cmp=pycaffe_intel_cori_netcdf

#todo: put a git clone in here
rm -rf ${cmp}
cp -r caffe ${cmp}
cd ${cmp}
cp Makefile.config.cori Makefile.config
#point to intel modules

maindir=/usr/common/software/python/2.7-anaconda/envs/caffe


export PATH=$maindir/bin:$PATH

make clean
make all -j24
#make test
#make runtest
make pycaffe

cd ..
