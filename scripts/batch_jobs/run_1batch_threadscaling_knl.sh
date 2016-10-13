#!/bin/bash
#SBATCH --ntasks-per-core=4
#SBATCH -N 1
#SBATCH -pdebug_knl
#SBATCH -C quad,flat

export OMP_PLACES=cores"(68)"
export OMP_PROC_BIND=spread

#execdir
execdir=/project/projectdirs/mpccc/tmalas/intelcaffe/install_cori-knl/bin

for th in 136 {128..16..-16} 48 32 24 16 8 4 2 1 
do
  export OMP_NUM_THREADS=$th
  exe=srun -n 1 -c 272 --cpu_bind=cores ${execdir}/caffe time -model=train_val.prototxt  -iterations=10
  echo $exe
  ${exe} 2>&1 | tee atlas_1batch_knl_${OMP_NUM_THREADS}th.out
done
