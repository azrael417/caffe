#!/bin/bash
#SBATCH --ntasks-per-core=4
#SBATCH -N 1
#SBATCH -pdebug_knl
#SBATCH -C quad,flat

module switch intel intel/17.0.0.098 

export OMP_PLACES=cores"(68)"
export OMP_PROC_BIND=spread
export OMP_NUM_THREADS=68

execdir=/project/projectdirs/mpccc/tmalas/intelcaffe/install_cori-knl/bin

results_dir=batch_scaling_knl_${OMP_NUM_THREADS}th
mkdir -p ${results_dir}

for bats in 1 2 4 8 16 32 64 128
do
  export BATCH_SIZE=$bats
  envsubst < train_val.prototxt.template > subst_train_val.prototxt
  exe="srun -n 1 -c 272 --cpu_bind=cores ${execdir}/caffe time -model=subst_train_val.prototxt  -iterations=10"
  export out_file=${results_dir}/atlas_${BATCH_SIZE}batch_knl_${OMP_NUM_THREADS}th.out
  echo $exe  | tee    $out_file
  $exe 2>&1 | tee -a $out_file
done
