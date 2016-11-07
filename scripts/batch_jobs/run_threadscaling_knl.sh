#!/bin/bash
#SBATCH --ntasks-per-core=4
#SBATCH -N 1
#SBATCH -pdebug_knl
#SBATCH -C quad,flat

CAFFE_ROOT=/project/projectdirs/mpccc/tmalas/intelcaffe/src/

module switch intel intel/16.0.3.210.test

export OMP_PLACES=cores"(68)"
export OMP_PROC_BIND=spread

export BATCH_SIZE=64
envsubst < train_val.prototxt.template > subst_train_val.prototxt

execdir=/project/projectdirs/mpccc/tmalas/intelcaffe/install_cori-knl/bin
results_dir=thread_scaling_${BATCH_SIZE}batch_knl
mkdir -p ${results_dir}

for th in 136 {128..16..-16} 48 32 24 16 8 4 2 1 
do
  export OMP_NUM_THREADS=$th
  export out_file=${results_dir}/atlas_${BATCH_SIZE}batch_knl_th${OMP_NUM_THREADS}.out
  exe="srun -n 1 -c 272 --cpu_bind=cores ${execdir}/caffe time -model=subst_train_val.prototxt  -iterations=10"
  echo $exe
  ${exe} 2>&1 | tee ${results_dir}/atlas_1batch_knl_${OMP_NUM_THREADS}th.out
done

source ${CAFFE_ROOT}scripts/batch_jobs/context.sh | tee $out_file
