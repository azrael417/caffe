#!/bin/bash
#SBATCH --ntasks-per-core=4
#SBATCH -N 1
#SBATCH -pdebug_knl
#SBATCH -C quad,flat

export OMP_NUM_THREADS=136
export OMP_PLACES=cores"(68)"
export OMP_PROC_BIND=spread

#execdir
execdir=/project/projectdirs/mpccc/tmalas/intelcaffe/install_cori-knl/bin

exe="srun -n 1 -c 272 --cpu_bind=cores ${execdir}/caffe time -model=train_val.prototxt"
echo $exe
${exe}
