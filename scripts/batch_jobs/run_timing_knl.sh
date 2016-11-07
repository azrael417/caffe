#!/bin/bash
#SBATCH --ntasks-per-core=4
#SBATCH -N 1
#SBATCH -pdebug_knl
#SBATCH -C quad,flat


module switch intel intel/16.0.3.210.test

export OMP_NUM_THREADS=136
export OMP_PLACES=cores"(68)"
export OMP_PROC_BIND=spread

CAFFE_ROOT=/project/projectdirs/mpccc/tmalas/intelcaffe/src/
source ${CAFFE_ROOT}scripts/batch_jobs/context.sh

execdir=/project/projectdirs/mpccc/tmalas/intelcaffe/install_cori-knl/bin

exe="srun -n 1 -c 272 --cpu_bind=cores ${execdir}/caffe time -model=train_val.prototxt -iterations 10"
echo $exe
${exe}
