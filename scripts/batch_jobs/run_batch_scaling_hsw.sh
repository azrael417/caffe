#!/bin/bash
#SBATCH --ntasks-per-core=1
#SBATCH -p debug
#SBATCH -N 1

export OMP_PLACES=threads
export OMP_PROC_BIND=spread

#executable
execdir=/project/projectdirs/mpccc/tmalas/intelcaffe/install_cori-hsw/bin/

export OMP_NUM_THREADS=16
for bats in 1 2 4 8 16 32 64 128
do
  export BATCH_SIZE=$bats
  envsubst < train_val.prototxt.template > subst_train_val.prototxt
  exe="srun -n 1 -c 32 --cpu_bind=socket -m block:cyclic ${execdir}/caffe time -model=subst_train_val.prototxt -iterations=10"
  echo $exe | tee    atlas_${BATCH_SIZE}batch_hsw_th${OMP_NUM_THREADS}.out
  $exe 2>&1 | tee -a atlas_${BATCH_SIZE}batch_hsw_th${OMP_NUM_THREADS}.out
done
