#!/bin/bash
#SBATCH --ntasks-per-core=1
#SBATCH -p debug
#SBATCH -N 1


export OMP_PLACES=threads
export OMP_PROC_BIND=spread

export BATCH_SIZE=1
envsubst < train_val.prototxt.template > subst_train_val.prototxt

execdir=/project/projectdirs/mpccc/tmalas/intelcaffe/install_cori-hsw/bin
results_dir=thread_scaling_${BATCH_SIZE}batch_hsw
mkdir -p ${results_dir}

for th in 32 24 16 8 4 2 1
do
  export OMP_NUM_THREADS=$th
  export out_file=${results_dir}/atlas_${BATCH_SIZE}batch_hsw_th${OMP_NUM_THREADS}.out
  exe="srun -n 1 -c 32 --cpu_bind=cores -m block:cyclic ${execdir}/caffe time -model=subst_train_val.prototxt -iterations=10"
  echo $exe | tee $out_file
  ${exe} 2>&1 | tee -a $out_file
done
