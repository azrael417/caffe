#!/bin/bash
#SBATCH --ntasks-per-core=1
#SBATCH -p debug
#SBATCH -N 1

export OMP_NUM_THREADS=16
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

module load sde
#executable
execdir=/project/projectdirs/mpccc/tmalas/intelcaffe/install_cori-hsw/bin/

exe="srun -n 1 -c 16 --cpu_bind=socket -m block:cyclic  sde -hsw -d -iform 1 -omix sde_hsw.out -i -global_region -- ${execdir}/caffe time -model=train_val.prototxt -iterations=10"
echo $exe
${exe}
