#!/bin/bash

# IntelCaffe repo. has to be located under the CAFFE_ROOT directory, using the name 'src'
CAFFE_ROOT='/project/projectdirs/mpccc/tmalas/intelcaffe'

module load cmake
module unload intel
module load intel/17.0.0.098
#module load intel
source /opt/intel/compilers_and_libraries/linux/mpi/intel64/bin/mpivars.sh intel64

#clean env
#module unload PrgEnv-gnu
#module unload PrgEnv-cray
#module unload PrgEnv-intel

#src directory
intelcaffe_version="0.9999.vtune"

#cd src
#git checkout tags/v${intelcaffe_version}
#cd ..

#load all required modules
#module load PrgEnv-intel
module load curl/7.48.0
module load netcdf/4.4.1
#load cmake
#module load cmake
#module load boost
#module load protobuf/2.4.1
#module load gflags
#module load glog
#module load cray-hdf5-parallel
#module load opencv
#module load lmdb
#module load snappy
#module load leveldb
#module load python
#module load vtune
#module load sde

#compiler flag:
cmp=intel_carl

#check out right version
cd src
#git checkout master
#git checkout netcdf-layer
cd ..

#rm -rf ${cmp}
cp -r src ${cmp}
cd ${cmp}

#point to intel modules
boost_dir=/usr/common/software/boost/1.61/hsw/intel
gflags_dir=/usr/common/software/gflags/2.1.2/hsw/intel
glog_dir=/usr/common/software/glog/0.3.4/hsw/intel
protobuf_dir=/usr/common/software/protobuf/2.4.1/intel
lmdb_dir=/usr/common/software/lmdb/0.9.18/intel
leveldb_dir=/usr/common/software/leveldb/1.18/intel
snappy_dir=/usr/common/software/snappy/1.1.3/intel
hdf5_dir=/usr/common/software/hdf5-parallel/1.8.16/intel
netcdf_dir=/usr/common/software/netcdf/4.4.1/hsw/intel
curl_dir=/usr/common/software/curl/7.48.0/hsw
openssl_dir=/usr/common/software/openssl/0.9.8

mkl_dnn_dir=${CAFFE_ROOT}"/src/external/mkl/mklml_lnx_2017.0.0.20160801"

#hdf5 stuff
export HDF5_DIR=${hdf5_dir}
export HDF5_ROOT=${hdf5_dir}
export HDF5_INCLUDE_OPTS=${hdf5_dir}/include
#gflags
export LD_LIBRARY_PATH=${gflags_dir}/lib:${LD_LIBRARY_PATH}
export PATH=${gflags_dir}/bin:${PATH}

INSTALL_PREFIX=${CAFFE_ROOT}/install_carl

#configure
cmake -G "Unix Makefiles" \
        -DBUILD_SHARED_LIBS=ON \
        -DBUILD_python=ON \
        -DBUILD_python_layer=ON \
        -DBoost_DIR=${boost_dir} \
        -DBoost_INCLUDE_DIR=${boost_dir}/include \
        -DCMAKE_CXX_COMPILER="mpiicpc" \
        -DCMAKE_CXX_FLAGS="-g -O3 -std=c++11 -mkl -xMIC-AVX512 -I${mkl_dnn_dir}/include" \
        -DCMAKE_C_COMPILER="mpiicc" \
        -DCMAKE_C_FLAGS="-g -O3 -std=c99 -mkl -xMIC-AVX512 -I${mkl_dnn_dir}/include" \
        -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
        -DCMAKE_LINKER="mpiicpc" \
        -DCPU_ONLY=ON \
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
        -DLevelDB_INCLUDE=${leveldb_dir}/include \
        -DLevelDB_LIBRARY=${leveldb_dir}/lib/libleveldb.so \
        -DSnappy_INCLUDE_DIR=${snappy_dir}/include \
        -DSnappy_LIBRARIES=${snappy_dir}/lib/libsnappy.so \
        -DHDF5_DIR=${hdf5_dir} \
        -DHDF5_IS_PARALLEL=ON \
        -DHDF5_hdf5_LIBRARY_RELEASE=${hdf5_dir}/lib/libhdf5.a \
        -DHDF5_hdf5_hl_LIBRARY_RELEASE=${hdf5_dir}/lib/libhdf5_hl.a \
        -DUSE_CUDNN=OFF \
        -DUSE_MKL2017_AS_DEFAULT_ENGINE=ON \
        -DBLAS=mkl \
        -DUSE_LEVELDB=OFF \
        -DUSE_LMDB=OFF \
        -DUSE_OPENCV=OFF \
        -DUSE_OPENMP=ON \
        -DUSE_NETCDF=ON \
        -DCMAKE_VERBOSE_MAKEFILE=ON \
        -DNETCDF_INCLUDE_DIR=${netcdf_dir}/include \
        -DNETCDF_LIBRARIES=${netcdf_dir}/lib/libnetcdf.a \
        .

#-DCURL_LIBRARIES=${curl_dir}/lib/libcurl.a \
#-DOPENSSL_LIBRARIES=${openssl_dir}/lib/libssl.a \


    #build
    make -j10 #2>&1 | tee ${INSTALL_PREFIX}/make_log.txt
    make install #2>&1 | tee -a ${INSTALL_PREFIX}/make_log.txt

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

