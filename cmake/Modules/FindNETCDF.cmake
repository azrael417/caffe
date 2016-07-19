# Try to find the NETCDF libraries and headers
#  NETCDF - system has NETCDF lib
#  NETCDF_INCLUDE_DIR - the NETCDF include directory
#  NETCDF_LIBRARIES - Libraries needed to use NETCDF

# FindCWD based on FindGMP by:
# Copyright (c) 2006, Laurent Montel, <montel@kde.org>
#
# Redistribution and use is allowed according to the terms of the BSD license.

# Adapted from FindCWD by:
# Copyright 2013 Conrad Steenberg <conrad.steenberg@gmail.com>
# Aug 31, 2013

find_path(NETCDF_INCLUDE_DIR NAMES  netcdf.h PATHS "$ENV{NETCDF_DIR}/include" DOC "Path in which the file netcdf/netcdf.h is located.")
find_library(NETCDF_LIBRARIES NAMES netcdf   PATHS "$ENV{NETCDF_DIR}/lib" DOC "Path in which the netcdf library is located.")
find_library(CURL_LIBRARIES NAMES curl   PATHS "$ENV{CURL_DIR}/lib" DOC "Path in which the curl library is located, needed by netcdf.")
find_library(OPENSSL_LIBRARIES NAMES ssl   PATHS "$ENV{CURL_DIR}/lib" DOC "Path in which the openssl library is located, needed by netcdf.")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NETCDF DEFAULT_MSG NETCDF_INCLUDE_DIR NETCDF_LIBRARIES)

if(NETCDF_FOUND)
  message(STATUS "Found netCDF    (include: ${NETCDF_INCLUDE_DIR}, library: ${NETCDF_LIBRARIES})")
  mark_as_advanced(NETCDF_INCLUDE_DIR NETCDF_LIBRARIES)

  #caffe_parse_header(${NETCDF_INCLUDE_DIR}/netcdf.h
  #                   NETCDF_VERSION_LINES NETCDF_VERSION_MAJOR NETCDF_VERSION_MINOR )
  #set(NETCDF_VERSION "${MDB_VERSION_MAJOR}.${MDB_VERSION_MINOR}.${MDB_VERSION_PATCH}")
endif()
