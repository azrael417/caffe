#!/bin/bash
echo "Userspace            : " `cat /sys/devices/system/cpu/cpu1/cpufreq/scaling_governor`
echo "Cur frequency        : " `cat /sys/devices/system/cpu/cpu1/cpufreq/scaling_cur_freq`
echo "Min frequency        : " `cat /sys/devices/system/cpu/cpu1/cpufreq/scaling_min_freq`
echo "Max frequency        : " `cat /sys/devices/system/cpu/cpu1/cpufreq/scaling_max_freq`
echo "Threads used     : " `echo $OMP_NUM_THREADS`
echo "KMP affinity     : " `echo $KMP_AFFINITY`
echo "KMP placethd     : " `echo $KMP_PLACE_THREADS`

echo "OMP PLACES     : " `echo $OMP_PLACES`
echo "OMP PROC BIND     : " `echo $OMP_PROC_BIND`

echo "Start env dump =================================================="
printenv
echo "End env dump =================================================="

echo "Intel Caffe version  : " `$CAFFE_ROOT/build/tools/caffe --version`
echo "MKL build date       : " `grep __INTEL_MKL_BUILD_DATE $CAFFE_ROOT/external/mkl/mklml\_lnx\_2017.0.0.20160801/include/mkl\_version.h`
