#!/bin/bash

#. ../environment.sh

#clean env
#module unload PrgEnv-gnu
#module unload PrgEnv-cray
#module unload PrgEnv-intel

#src directory
intelcaffe_version="0.9999_mkl"

#cd src
#git checkout tags/v${intelcaffe_version}
#cd ..

#load all required modules
module swap craype-mic-knl craype-haswell
module load PrgEnv-intel
module load cmake/3.5.2
module load intel/17.0.0.098
module load cray-memkind
#source /opt/intel/impi/5.1.3.210/bin64/mpivars.sh

#load cmake
#module load cmake
module load boost
module load protobuf/2.4.1
module load gflags
module load glog
module load cray-hdf5-parallel/1.8.16
module load opencv
module load lmdb
module load snappy
module load leveldb
module load python
module load netcdf/4.4.1

#compiler flag:
cmp=intel

#check out right version
#cd src
#git checkout master
#cd ..

rm -rf ${cmp}_cori-hsw
cp -r src ${cmp}_cori-hsw
cd ${cmp}_cori-hsw

#get directory paths
boost_dir=$(module show boost/1.61 2>&1 > /dev/null | grep BOOST_DIR | awk '{print $3}' | sed 's|/usr/common/software|/global/common/cori/software|g')

#boost_dir="/global/cscratch1/sd/swowner/boost/1_61_0-GNU/hsw/gnu"
gflags_dir=$(module show gflags 2>&1 > /dev/null | grep LD_LIBRARY_PATH | awk '{split($3,a,"/lib"); print a[1]}' | sed 's|/usr/common/software|/global/common/cori/software|g')
glog_dir=$(module show glog 2>&1 > /dev/null | grep LD_LIBRARY_PATH | awk '{split($3,a,"/lib"); print a[1]}' | sed 's|/usr/common/software|/global/common/cori/software|g')
protobuf_dir=$(module show protobuf/2.4.1 2>&1 > /dev/null | grep LD_LIBRARY_PATH | awk '{split($3,a,"/lib"); print a[1]}' | sed 's|/usr/common/software|/global/common/cori/software|g')
lmdb_dir=$(module show lmdb 2>&1 > /dev/null | grep LD_LIBRARY_PATH | awk '{split($3,a,"/lib"); print a[1]}' | sed 's|/usr/common/software|/global/common/cori/software|g')
leveldb_dir=$(module show leveldb 2>&1 > /dev/null | grep LD_LIBRARY_PATH | awk '{split($3,a,"/lib"); print a[1]}' | sed 's|/usr/common/software|/global/common/cori/software|g')
snappy_dir=$(module show snappy 2>&1 > /dev/null | grep LD_LIBRARY_PATH | awk '{split($3,a,"/lib"); print a[1]}' | sed 's|/usr/common/software|/global/common/cori/software|g')
netcdf_dir=$(module show netcdf/4.4.1 2>&1 > /dev/null | grep LD_LIBRARY_PATH | awk '{split($3,a,"/lib"); print a[1]}' | sed 's|/usr/common/software|/global/common/cori/software|g')
mkl_dir=$(echo ${MKLROOT})

#-DAtlas_BLAS_LIBRARY=${mkl_dir}/lib/intel64/libmkl_core.a \
#-DAtlas_CBLAS_INCLUDE_DIR=${mkl_dir}/include \
#-DAtlas_CLAPACK_INCLUDE_DIR=${mkl_dir}/include \
#-DAtlas_LAPACK_LIBRARY=${mkl_dir}/lib/libmkl_core.so \
#-DAtlas_CBLAS_LIBRARY=${mkl_dir}/lib/libmkl_core.so \

CC=/opt/cray/pe/craype/2.5.5/bin/cc
CXX=/opt/cray/pe/craype/2.5.5/bin/CC
LDFLAGS="-Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -liomp5 -lpthread -lm -ldl -L${netcdf_dir}/lib -lnetcdf -lmemkind"

export CRAYPE_LINK_TYPE=dynamic

#configure
cmake -G "Unix Makefiles" \
        -DCMAKE_VERBOSE_MAKEFILE=ON \
        -DBUILD_SHARED_LIBS=ON \
        -DBUILD_python=ON \
        -DBUILD_python_layer=ON \
        -DBoost_DIR=${boost_dir} \
        -DBoost_INCLUDE_DIR=${boost_dir}/include \
        -DBoost_ATOMIC_LIBRARY_DEBUG=${boost_dir}/lib/libboost_atomic.so \
        -DBoost_ATOMIC_LIBRARY_RELEASE=${boost_dir}/lib/libboost_atomic.so \
        -DBoost_CHRONO_LIBRARY_DEBUG=${boost_dir}/lib/libboost_chrono.so \
        -DBoost_CHRONO_LIBRARY_RELEASE=${boost_dir}/lib/libboost_chrono.so \
        -DBoost_DATE_TIME_LIBRARY_DEBUG=${boost_dir}/lib/libboost_date_time.so \
        -DBoost_DATE_TIME_LIBRARY_RELEASE=${boost_dir}/lib/libboost_date_time.so \
        -DBoost_FILESYSTEM_LIBRARY_DEBUG=${boost_dir}/lib/libboost_filesystem.so \
        -DBoost_FILESYSTEM_LIBRARY_RELEASE=${boost_dir}/lib/libboost_filesystem.so \
        -DBoost_LIBRARY_DIR=${boost_dir}/intel/lib \
        -DBoost_PYTHON_LIBRARY_DEBUG=${boost_dir}/lib/libboost_python.so \
        -DBoost_PYTHON_LIBRARY_RELEASE=${boost_dir}/lib/libboost_python.so \
        -DBoost_SYSTEM_LIBRARY_DEBUG=${boost_dir}/lib/libboost_system.so \
        -DBoost_SYSTEM_LIBRARY_RELEASE=${boost_dir}/lib/libboost_system.so \
        -DBoost_THREAD_LIBRARY_DEBUG=${boost_dir}/lib/libboost_thread.so \
        -DBoost_THREAD_LIBRARY_RELEASE=${boost_dir}/lib/libboost_thread.so \
        -DCMAKE_CXX_COMPILER="${CXX}" \
        -DCMAKE_CXX_FLAGS="-g -O3 -std=c++11 -mkl -xCORE-AVX2" \
        -DCMAKE_C_COMPILER="${CC}" \
        -DCMAKE_C_FLAGS="-g -O3 -std=c99 -mkl -xCORE-AVX2" \
        -DCMAKE_INSTALL_PREFIX="/project/projectdirs/mpccc/tkurth/NESAP/intelcaffe/install_cori-hsw" \
        -DCMAKE_LINKER="${CXX}" \
        -DCMAKE_SHARED_LINKER_FLAGS="${LDFLAGS}" \
        -DCMAKE_MODULE_LINKER_FLAGS="${LDFLAGS}" \
        -DCPU_ONLY=ON \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DGFLAGS_ROOT_DIR=${gflags_dir} \
        -DGFLAGS_LIBRARY=${gflags_dir}/lib/libgflags.so \
        -DGFLAGS_INCLUDE_DIR=${gflags_dir}/include \
        -DGLOG_ROOT_DIR=${glog_dir} \
        -DGLOG_INCLUDE_DIR=${glog_dir}/include \
        -DOpenMP_CXX_FLAGS="-qopenmp" \
        -DOpenMP_C_FLAGS="-qopenmp" \
        -DPROTOBUF_INCLUDE_DIR=${protobuf_dir}/include \
        -DPROTOBUF_LIBRARY=${protobuf_dir}/lib/libprotobuf.a \
        -DPROTOBUF_LITE_LIBRARY=${protobuf_dir}/lib/libprotobuf-lite.a \
        -DPROTOBUF_PROTOC_EXECUTABLE=${protobuf_dir}/bin/protoc \
        -DPROTOBUF_PROTOC_LIBRARY=${protobuf_dir}/lib/libprotoc.a \
        -DLMDB_INCLUDE_DIR=${lmdb_dir}/include \
        -DLMDB_LIBRARIES=${lmdb_dir}/lib/liblmdb.so \
        -DHDF5_DIR=${HDF5_DIR} \
        -DHDF5_HL_INCLUDE_DIR=${HDF5_DIR}/include \
        -DHDF5_hdf5_LIBRARY_RELEASE=${HDF5_DIR}/lib/libhdf5.a \
        -DHDF5_hdf5_hl_LIBRARY_RELEASE=${HDF5_DIR}/lib/libhdf5_hl.a \
        -DHDF5_IS_PARALLEL=ON \
        -DUSE_NETCDF=ON \
        -DNETCDF_INCLUDE_DIR=${netcdf_dir}/include \
        -DNETCDF_LIBRARIES=${netcdf_dir}/lib/libnetcdf.so \
        -DUSE_CUDNN=OFF \
        -DBLAS=mkl \
        -DUSE_MKL2017_AS_DEFAULT_ENGINE=ON \
        -DMKL_USE_SINGLE_DYNAMIC_LIBRARY=OFF \
        -DMisc_LIBRARIES="${LDFLAGS}" \
        -DUSE_LEVELDB=ON \
        -DLevelDB_INCLUDE=${leveldb_dir}/include \
        -DLevelDB_LIBRARY=${leveldb_dir}/lib/libleveldb.so \
        -DSnappy_INCLUDE_DIR=${snappy_dir}/include \
        -DSnappy_LIBRARIES=${snappy_dir}/lib/libsnappy.so \
        -DUSE_LMDB=ON \
        -DUSE_OPENCV=OFF \
        -DUSE_OPENMP=ON \
        .

    #build
    make -j10
    make install

cd ..
#variables
#export CRAYPE_LINK_TYPE=dynamic
#CMAKE_BUILD_TYPE="Debug"
#CMAKE_INSTALL_PREFIX=${SOFTWAREPATH}/llvm/3.8.0/${cmp}
#    LLVM_TARGETS_TO_BUILD="X86"
#
#    CXX="CC"
#    CXXFLAGS="-fPIE -std=c++11"
#    CC="cc"
#    CFLAGS="-fPIE -std=c99"
#    LDFLAGS="-pie"
#    
#    cmake -G "Unix Makefiles" \
#        -DCMAKE_CXX_COMPILER=${CXX} \
#        -DCMAKE_CXX_FLAGS=${CXXFLAGS} \
#        -DCMAKE_C_COMPILER=${CC} \
#        -DCMAKE_C_FLAGS=${CFLAGS} \
#        -DCMAKE_LD_FLAGS=${LDFLAGS} \
#        -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE \
#        -DCMAKE_INSTALL_PREFIX=$CMAKE_INSTALL_PREFIX \
#        -DLLVM_TARGETS_TO_BUILD=$LLVM_TARGETS_TO_BUILD \
#        $SRC
#
#    make -j10; make install
#    
#    cd ..
#    
#    #unload env
#    module unload PrgEnv-${cmp}
#done

