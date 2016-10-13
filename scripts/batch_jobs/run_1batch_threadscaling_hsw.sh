#!/bin/bash
#SBATCH --ntasks-per-core=1
#SBATCH -p debug
#SBATCH -N 1

export OMP_PLACES=threads
export OMP_PROC_BIND=spread

#executable
execdir=/project/projectdirs/mpccc/tmalas/intelcaffe/install_cori-hsw/bin/

for th in 32 24 16 8 4 2 1
do
  export OMP_NUM_THREADS=$th
  exe=srun -n 1 -c 32 --cpu_bind=cores -m block:cyclic ${execdir}/caffe time -model=train_val.prototxt -iterations=10
  echo $exe
  ${exe} 2>&1 | tee atlas_1batch_hsw_th${OMP_NUM_THREADS}.out
done
