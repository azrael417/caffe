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
module load PrgEnv-intel
module load cmake
module load intel/2017.beta.up2
source /opt/intel/impi/5.1.3.210/bin64/mpivars.sh

#load cmake
#module load cmake
module load boost
module load protobuf/2.4.1
module load gflags
module load glog
module load cray-hdf5-parallel
module load opencv
module load lmdb
module load snappy
module load leveldb
module load python
module load netcdf

#compiler flag:
cmp=intel

#check out right version
#cd src
#git checkout master
#cd ..

rm -rf ${cmp}_cori
cp -r src ${cmp}_cori
cd ${cmp}_cori

#get directory paths
#boost_dir=$(module show boost 2>&1 > /dev/null | grep BOOST_DIR | awk '{print $3}')
boost_dir="/usr/common/software/boost/1.59/intel"
gflags_dir=$(module show gflags 2>&1 > /dev/null | grep LD_LIBRARY_PATH | awk '{split($3,a,"/lib"); print a[1]}')
glog_dir=$(module show glog 2>&1 > /dev/null | grep LD_LIBRARY_PATH | awk '{split($3,a,"/lib"); print a[1]}')
protobuf_dir=$(module show protobuf/2.4.1 2>&1 > /dev/null | grep LD_LIBRARY_PATH | awk '{split($3,a,"/lib"); print a[1]}')
lmdb_dir=$(module show lmdb 2>&1 > /dev/null | grep LD_LIBRARY_PATH | awk '{split($3,a,"/lib"); print a[1]}')
leveldb_dir=$(module show leveldb 2>&1 > /dev/null | grep LD_LIBRARY_PATH | awk '{split($3,a,"/lib"); print a[1]}')
snappy_dir=$(module show snappy 2>&1 > /dev/null | grep LD_LIBRARY_PATH | awk '{split($3,a,"/lib"); print a[1]}')
netcdf_dir=$(module show netcdf 2>&1 > /dev/null | grep LD_LIBRARY_PATH | awk '{split($3,a,"/lib"); print a[1]}')
mkl_dir=${MKLROOT}

#-DAtlas_BLAS_LIBRARY=${mkl_dir}/lib/intel64/libmkl_core.a \
#-DAtlas_CBLAS_INCLUDE_DIR=${mkl_dir}/include \
#-DAtlas_CLAPACK_INCLUDE_DIR=${mkl_dir}/include \
#-DAtlas_LAPACK_LIBRARY=${mkl_dir}/lib/libmkl_core.so \
#-DAtlas_CBLAS_LIBRARY=${mkl_dir}/lib/libmkl_core.so \

CC=/opt/cray/pe/craype/2.5.5/bin/cc
CXX=/opt/cray/pe/craype/2.5.5/bin/CC

export CRAYPE_LINK_TYPE=dynamic

#configure
cmake -G "Unix Makefiles" \
        -DCMAKE_VERBOSE_MAKEFILE=ON \
        -DBUILD_SHARED_LIBS=ON \
        -DBUILD_python=ON \
        -DBUILD_python_layer=ON \
        -DBoost_DIR=${boost_dir} \
        -DBoost_INCLUDE_DIR=${boost_dir}/include \
        -DCMAKE_CXX_COMPILER="${CXX}" \
        -DCMAKE_CXX_FLAGS="-dynamic -fPIE -std=c++11 -mkl -xCORE-AVX2" \
        -DCMAKE_C_COMPILER="${CC}" \
        -DCMAKE_C_FLAGS="-dynamic -fPIE -std=c99 -mkl -xCORE-AVX2" \
        -DCMAKE_INSTALL_PREFIX="/project/projectdirs/mpccc/tkurth/NESAP/intelcaffe/install_cori" \
        -DCMAKE_LINKER="${CXX}" \
        -DCMAKE_EXE_LINKER_FLAGS="-shared -mkl" \
        -DCMAKE_MODULE_LINKER_FLAGS="-shared -mkl" \
        -DCPU_ONLY=ON \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DGFLAGS_ROOT_DIR=${gflags_dir} \
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
        -DUSE_NETCDF=ON \
        -DNETCDF_INCLUDE_DIR=${netcdf_dir}/include \
        -DNETCDF_LIBRARIES=${netcdf_dir}/lib/libnetcdf.so \
        -DUSE_CUDNN=OFF \
        -DBLAS=mkl \
        -DUSE_MKL2017_AS_DEFAULT_ENGINE=ON \
        -DUSE_LEVELDB=ON \
        -DLevelDB_INCLUDE=${leveldb_dir}/include \
        -DLevelDB_LIBRARY=${leveldb_dir}/lib/libleveldb.so \
        -DSnappy_INCLUDE_DIR=${snappy_dir}/include \
        -DSnappy_LIBRARIES=${snappy_dir}/lib/libsnappy.so \
        -DUSE_LMDB=ON \
        -DUSE_OPENCV=ON \
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

