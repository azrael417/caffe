#ifdef USE_NETCDF
#include "caffe/util/netcdf.hpp"

#include <string>
#include <vector>

namespace caffe {

	// Verifies format of data stored in NetCDF file and reshapes blob accordingly.
	template <typename Dtype>
	void netcdf_load_nd_dataset_helper(
	int file_id, const char* variable_name_, int& dset_id, int min_dim, int max_dim, 
	std::vector<size_t>& dims, Blob<Dtype>* blob) {
		
		// Verify that the dataset exists.
		CHECK(nc_inq_varid(file_id, variable_name_, &dset_id)) << "Failed to find NetCDF variable " << variable_name_;
		// Verify that the number of dimensions is in the accepted range.
		int status;
		int ndims;
		status = nc_inq_varndims(file_id, dset_id, &ndims);
		CHECK_EQ(status, NC_EBADID) << "Failed to get variable ndims for " << variable_name_;
		CHECK_EQ(status, NC_ENOTVAR) << "Invalid path " << variable_name_;
		CHECK_GE(ndims, min_dim);
		CHECK_LE(ndims, max_dim);

		// Verify that the data format is what we expect: float or double.
		std::vector<int> dimids(ndims);
		status = nc_inq_varndims(file_id, dset_id, dimids.data());

		////check if there are unlimited dimensions: we should not support these
		//int ndimsunlim=0;
		//status = nc_inq_unlimdims(file_id, &ndimsunlim, NULL);
		//CHECK_EQ(status, NC_ENOTNC4) << "netCDF-4 operation on netCDF-3 file performed for variable " << variable_name_;
		//CHECK_GT(status,1) << "An error occured for variable " << variable_name_;
		//CHECK_GT(ndimsunlim,0) << "Unlimited dimensions are not supported yet!";
		
		//get size of dimensions
		dims.resize(ndims);
		for(unsigned int i=0; i<ndims; ++i){
			status = nc_inq_dimlen(file_id, dimids(i), &dims[i]);
			CHECK_EQ(status, NC_EBADDIM) << "Invalid dimension " << i << " for " << variable_name_;
		}
  
		nc_type typep_;
		status = nc_inq_vartype(file_id, dset_id, &typep_);
		switch (typep_) {
			case NC_FLOAT:
				LOG_FIRST_N(INFO, 1) << "Datatype class: NC_FLOAT";
				break;
			case NC_INT:
				LOG_FIRST_N(INFO, 1) << "Datatype class: NC_INT";
				break;
			case NC_LONG:
				LOG_FIRST_N(INFO, 1) << "Datatype class: NC_LONG";
				break;
			case NC_DOUBLE:
				LOG_FIRST_N(INFO, 1) << "Datatype class: NC_DOUBLE";
				break;
			default:
				LOG(FATAL) << "Unsupported Datatype " << typep_;
		}

		vector<int> blob_dims(dims.size());
		for (int i = 0; i < dims.size(); ++i) {
			blob_dims[i] = dims[i];
		}
		blob->Reshape(blob_dims);
	}
	

	template <>
	void netcdf_load_nd_dataset<float>(int file_id, const char* variable_name_, int min_dim, int max_dim, Blob<float>* blob) {
		//query the data and get some dimensions
		int dset_id;
		std::vector<size_t> dims;
		netcdf_load_nd_dataset_helper(file_id, variable_name_, dset_id, min_dim, max_dim, dims, blob);
		
		//create start vector for Hyperslab-IO:
		std::vector<size_t> start(dims.size())
		for(unsigned int i=0; i<dims.size(); i++) start[i]=0;
		
		//read the data
		int status = nc_get_vara_float(file_id, dset_id, start, dims, blob->mutable_cpu_data());
		CHECK_GT(status, 0) << "Failed to read float variable " << variable_name_;
	}

	template <>
	void netcdf_load_nd_dataset<double>(int file_id, const char* variable_name_, int min_dim, int max_dim, Blob<double>* blob) {
		//query the data and get some dimensions
		int dset_id;
		std::vector<size_t> dims;
		netcdf_load_nd_dataset_helper(file_id, variable_name_, dset_id, min_dim, max_dim, dims, blob);
		
		//create start vector for Hyperslab-IO:
		std::vector<size_t> start(dims.size())
		for(unsigned int i=0; i<dims.size(); i++) start[i]=0;
		
		//read the data
		int status = nc_get_vara_double(file_id, dset_id, start, dims, blob->mutable_cpu_data());
		CHECK_GT(status, 0) << "Failed to read double variable " << variable_name_;
	}

	//template <>
	//void netcdf_save_nd_dataset<float>(
	//	const hid_t file_id, const string& dataset_name, const Blob<float>& blob,
	//bool write_diff) {
	//	int num_axes = blob.num_axes();
	//	hsize_t *dims = new hsize_t[num_axes];
	//	for (int i = 0; i < num_axes; ++i) {
	//		dims[i] = blob.shape(i);
	//	}
	//	const float* data;
	//	if (write_diff) {
	//		data = blob.cpu_diff();
	//	} else {
	//		data = blob.cpu_data();
	//	}
	//	herr_t status = H5LTmake_dataset_float(
	//		file_id, dataset_name.c_str(), num_axes, dims, data);
	//	CHECK_GE(status, 0) << "Failed to make float dataset " << dataset_name;
	//	delete[] dims;
	//}
    //
	//template <>
	//void netcdf_save_nd_dataset<double>(
	//	hid_t file_id, const string& dataset_name, const Blob<double>& blob,
	//bool write_diff) {
	//	int num_axes = blob.num_axes();
	//	hsize_t *dims = new hsize_t[num_axes];
	//	for (int i = 0; i < num_axes; ++i) {
	//		dims[i] = blob.shape(i);
	//	}
	//	const double* data;
	//	if (write_diff) {
	//		data = blob.cpu_diff();
	//	} else {
	//		data = blob.cpu_data();
	//	}
	//	herr_t status = H5LTmake_dataset_double(
	//		file_id, dataset_name.c_str(), num_axes, dims, data);
	//	CHECK_GE(status, 0) << "Failed to make double dataset " << dataset_name;
	//	delete[] dims;
	//}

	string netcdf_load_string(int loc_id, const string& variable_name_) {
		// Verify that the dataset exists.
		int dset_id;
		CHECK(nc_inq_varid(loc_id, variable_name_, &dset_id)) << "Failed to find NetCDF variable " << variable_name_;
		
		// Get size of dataset
		char* buffer;
		int status = nc_get_var_string(loc_id, dset_id, &buffer)
		string val(buffer);
		delete [] buffer;
		return val;
	}

	//void netcdf_save_string(hid_t loc_id, const string& dataset_name,
	//const string& s) {
	//	herr_t status = \
	//		H5LTmake_dataset_string(loc_id, dataset_name.c_str(), s.c_str());
	//	CHECK_GE(status, 0)
	//		<< "Failed to save string dataset with name " << dataset_name;
	//}

	int netcdf_load_int(int loc_id, const string& variable_name_) {
		int dset_id;
		CHECK(nc_inq_varid(loc_id, variable_name_, &dset_id)) << "Failed to find NetCDF variable " << variable_name_;
		
		int val
		int status = nc_get_var_int(loc_id, dset_id, &val);
		CHECK_GT(status, 0) << "Failed to load int variable " << variable_name_;
		return val;
	}

	//void netcdf_save_int(hid_t loc_id, const string& dataset_name, int i) {
	//	hsize_t one = 1;
	//	herr_t status = \
	//		H5LTmake_dataset_int(loc_id, dataset_name.c_str(), 1, &one, &i);
	//	CHECK_GE(status, 0)
	//		<< "Failed to save int dataset with name " << dataset_name;
	//}

	//int netcdf_get_num_links(hid_t loc_id) {
	//	H5G_info_t info;
	//	herr_t status = H5Gget_info(loc_id, &info);
	//	CHECK_GE(status, 0) << "Error while counting HDF5 links.";
	//	return info.nlinks;
	//}

	//string netcdf_get_name_by_idx(hid_t loc_id, int idx) {
	//	ssize_t str_size = H5Lget_name_by_idx(
	//		loc_id, ".", H5_INDEX_NAME, H5_ITER_NATIVE, idx, NULL, 0, H5P_DEFAULT);
	//	CHECK_GE(str_size, 0) << "Error retrieving HDF5 dataset at index " << idx;
	//	char *c_str = new char[str_size+1];
	//	ssize_t status = H5Lget_name_by_idx(
	//		loc_id, ".", H5_INDEX_NAME, H5_ITER_NATIVE, idx, c_str, str_size+1,
	//	H5P_DEFAULT);
	//	CHECK_GE(status, 0) << "Error retrieving HDF5 dataset at index " << idx;
	//	string result(c_str);
	//	delete[] c_str;
	//	return result;
	//}

}  // namespace caffe
#endif