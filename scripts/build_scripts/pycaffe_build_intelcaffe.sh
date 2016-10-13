#!/bin/bash -l

module load PrgEnv-intel/6.0.3
module swap intel intel/2017.beta.up2
source /opt/intel/compilers_and_libraries/linux/mpi/intel64/bin/mpivars.sh intel64
module unload intel 
module load intel/2017.beta.up2
module load curl/7.48.0
cmp=pycaffe_intel_cori


rm -rf ${cmp}
cp -r caffe ${cmp}
cd ${cmp}
cp Makefile.config.cori Makefile.config
#point to intel modules

maindir=/usr/common/software/python/2.7-anaconda/envs/caffe_env


export PATH=$maindir/bin:$PATH


make all -j24
#make test
#make runtest
make pycaffe

cd ..
