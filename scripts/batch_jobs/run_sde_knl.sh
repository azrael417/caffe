#!/bin/bash
#SBATCH --ntasks-per-core=4
#SBATCH -N 1
#SBATCH -pdebug_knl
#SBATCH -C quad,flat

CAFFE_ROOT=/project/projectdirs/mpccc/tmalas/intelcaffe/src/
MODEL=train_val.prototxt
SDE_PARSER="/global/homes/t/tmalas/ai_scripts/parse-sde.sh" # available here: http://portal.nersc.gov/project/m888/examples/stream-ai-example_160518.tgz
module unload intel

# CORI/GERTY
#module load intel/16.0.3.210.test
#export OMP_PLACES=cores"(68)"
#export OMP_NUM_THREADS=68
#execdir=/project/projectdirs/mpccc/tmalas/intelcaffe/install_cori-knl/bin
#exe_pre="srun -n 1 -c 272 --cpu_bind=cores "

# CARL
module load intel/17.0.0.098
export OMP_PLACES=cores"(64)"
export OMP_NUM_THREADS=64
execdir=/project/projectdirs/mpccc/tmalas/intelcaffe/install_carl/bin
exe_pre=""

module load sde

export OMP_PROC_BIND=spread

results_dir=sde_knl
mkdir -p ${results_dir}

# Get the layers count in the prototxt file
export layers_count="$(cat ${MODEL} | grep layer | wc -l)"

# run once to get the layers time
export out_file=${results_dir}/knl_time.out
if [ ! -f ${out_file} ]; then
  exe=${exe_pre}"${execdir}/caffe time -model=${MODEL} -iterations=100"
  echo $exe  | tee ${out_file}
  ${exe} 2>&1 | tee -a ${out_file}
fi
 
export OMP_NUM_THREADS=16
for direction in 1 0
do
  for layer in $(seq 0 `expr $layers_count - 1`)
  do
    export out_file=${results_dir}/knl_sde_layer${layer}_dir${direction}.out
    if [ ! -f ${out_file} ]; then
      exe=${exe_pre}"sde64 -knl -d -iform 1 -omix ${out_file}.mix -global_region -start_ssc_mark 111:repeat -stop_ssc_mark 222:repeat  -- ${execdir}/caffe time -model=${MODEL} -iterations=1 -prof_layer=${layer} -prof_forward_direction=${direction}"
      echo $exe   | tee ${out_file}
      ${exe} 2>&1 | tee -a ${out_file}
      ${SDE_PARSER} ${out_file}.mix | tee -a ${out_file}
    fi
  done
done

source ${CAFFE_ROOT}scripts/batch_jobs/context.sh > ${results_dir}/context.txt

