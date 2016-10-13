#!/bin/bash
#SBATCH --ntasks-per-core=4
#SBATCH -N 1
#SBATCH -pdebug_knl
#SBATCH -C quad,flat

export OMP_PLACES=cores"(68)"
export OMP_PROC_BIND=spread

#execdir
execdir=/project/projectdirs/mpccc/tmalas/intelcaffe/install_cori-knl/bin

export OMP_NUM_THREADS=136

for bats in 1 2 4 8 16 32 64 128
do
  export BATCH_SIZE=$bats
  envsubst < train_val.prototxt.template > subst_train_val.prototxt
  exe="srun -n 1 -c 272 --cpu_bind=cores ${execdir}/caffe time -model=subst_train_val.prototxt  -iterations=10"
  echo $exe  | tee    atlas_${BATCH_SIZE}batch_knl_${OMP_NUM_THREADS}th.out
  $exe  2>&1 | tee -a atlas_${BATCH_SIZE}batch_knl_${OMP_NUM_THREADS}th.out
done
