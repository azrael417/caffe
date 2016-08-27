#!/bin/bash -l

. ./environment.sh
module swap intel intel/2017.beta.up2
source /opt/intel/compilers_and_libraries/linux/mpi/intel64/bin/mpivars.sh intel64


intelcaffe_version="0.9999.vtune"

module unload intel 
module load intel/2017.beta.up2
module load curl/7.48.0
#module load netcdf/4.4.1
cmp=pycaffe_intel


rm -rf ${cmp}
cp -r caffe ${cmp}
cd ${cmp}

#point to intel modules

maindir=/usr/common/software/python/2.7-anaconda/envs/caffe_env


export PATH=$maindir/bin:$PATH


make all -j8
make test
make runtest
make pycaffe

cd ..
