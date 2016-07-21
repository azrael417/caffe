#ifdef USE_NETCDF
#include "caffe/util/netcdf.hpp"

#include <string>
#include <vector>

namespace caffe {

	void netcdf_check_variable_helper(const int& file_id, const string& variable_name_, int& dset_id, const int& min_dim, const int& max_dim, std::vector<size_t>& dims){
		//look up variable
		int status;
		status=nc_inq_varid(file_id, variable_name_.c_str(), &dset_id);
		CHECK(status != NC_EBADID) << "Bad NetCDF File-ID specified";
		
		//CHECK() << "Failed to find NetCDF variable " << variable_name_;
		// Verify that the number of dimensions is in the accepted range.
		int ndims;
		status = nc_inq_varndims(file_id, dset_id, &ndims);
		if(status != 0){
			CHECK(status != NC_EBADID) << "Failed to get variable ndims for " << variable_name_;
			CHECK(status != NC_ENOTVAR) << "Invalid path " << variable_name_;
		}
		CHECK_GE(ndims, min_dim);
		CHECK_LE(ndims, max_dim);

		// Verify that the data format is what we expect: float or double.
		std::cout << "Number of dimensions: " << ndims << std::endl;
		std::vector<int> dimids(ndims);
		status = nc_inq_vardimid(file_id, dset_id, dimids.data());

		//query unlimited dimensions
		//int ndims_unlimited;
		//std::vector<int> dimids_unlimited;
		//status = nc_inq_unlimdims(file_id, &ndims_unlimited, dimids_unlimited.data());
		//CHECK(status != NC_ENOTNC4) << "netCDF-4 operation on netCDF-3 file performed for variable " << variable_name_;
		//for(unsigned int i=0; i<dimids_unlimited.size(); i++){
		//	std::cout << i << " " << dimids_unlimited[i] << std::endl;
		//}
		//for(unsigned int i=0; i<dimids.size(); i++){
		//	std::cout << i << " " << dimids[i] << std::endl;
		//}

		//get size of dimensions
		dims.resize(ndims);
		for(unsigned int i=0; i<ndims; ++i){
			status = nc_inq_dimlen(file_id, dimids[i], &dims[i]);
			CHECK(status != NC_EBADDIM) << "Invalid dimension " << i << " for " << variable_name_;
		}

		nc_type typep_;
		status = nc_inq_vartype(file_id, dset_id, &typep_);
		switch (typep_) {
			case NC_FLOAT:
			LOG_FIRST_N(INFO, 1) << "Datatype class: NC_FLOAT";
			break;
			case NC_INT:
			LOG_FIRST_N(INFO, 1) << "Datatype class: NC_INT or NC_LONG";
			break;
			case NC_DOUBLE:
			LOG_FIRST_N(INFO, 1) << "Datatype class: NC_DOUBLE";
			break;
			default:
			LOG(FATAL) << "Unsupported Datatype " << typep_;
		}
	}
  
	// Verifies format of data stored in NetCDF file and reshapes blob accordingly.
	template <typename Dtype>
	void netcdf_load_nd_dataset_helper(const int& file_id, const std::vector<string>& netcdf_variables_, std::vector<int>& dset_ids, const int& min_dim, const int& max_dim, std::vector<size_t>& dims, Blob<Dtype>* blob) {
    
		// Obtain all sizes for variable 0:
		string variable_name_=netcdf_variables_[0];
		netcdf_check_variable_helper(file_id, variable_name_, dset_ids[0], min_dim, max_dim, dims);
    
		//make sure that all other dimensions for the other variables fit as well:
		for(unsigned int i=1; i<netcdf_variables_.size(); i++){
			std::vector<size_t> tmpdims;
			variable_name_=netcdf_variables_[i];
			netcdf_check_variable_helper(file_id, variable_name_, dset_ids[i], min_dim, max_dim, tmpdims);
			CHECK_EQ(tmpdims.size(),dims.size()) << "Number of dimensions of variable " << netcdf_variables_[0] << " and " << variable_name_ << " do not agree!";
			for(unsigned int d=0; d<tmpdims.size(); d++){
				CHECK_EQ(tmpdims[d],dims[d]) << "Dimension " << d << " does not agree for " << netcdf_variables_[0] << " and " << variable_name_;
			}
		}

		vector<int> blob_dims(dims.size()+1);
		for (int i = 0; i < dims.size(); ++i) {
			blob_dims[i] = dims[i];
		}
		blob_dims[dims.size()]=netcdf_variables_.size();
		
		//DEBUG
		for (int i = 0; i < dims.size()+1; ++i) std::cout << "blob_dims[" << i << "] = " << blob_dims[i] << std::endl;
		//DEBUG
		
		blob->Reshape(blob_dims);
	} 

	template <>
	void netcdf_load_nd_dataset<float>(const int& file_id, const std::vector<string>& netcdf_variables_, const int& min_dim, const int& max_dim, Blob<float>* blob) {
		//query the data and get some dimensions
		std::vector<int> dset_ids(netcdf_variables_.size());
		std::vector<size_t> dims;
		netcdf_load_nd_dataset_helper(file_id, netcdf_variables_, dset_ids, min_dim, max_dim, dims, blob);
		
		//create start vector for Hyperslab-IO:
		std::vector<size_t> start(dims.size());
		unsigned long offset=1;
		for(unsigned int i=0; i<dims.size(); i++){
			offset*=dims[i];
			start[i]=0;
		}
		
		//read the data
		for(unsigned int i=0; i<netcdf_variables_.size(); i++){	
			int status = nc_get_vara_float(file_id, dset_ids[i], start.data(), dims.data(), &(blob->mutable_cpu_data()[i*offset]));
			CHECK_GT(status, 0) << "Failed to read NetCDF float variable " << netcdf_variables_[i];
		}
	}

	template <>
	void netcdf_load_nd_dataset<double>(const int& file_id, const std::vector<string>& netcdf_variables_, const int& min_dim, const int& max_dim, Blob<double>* blob) {
		//query the data and get some dimensions
		std::vector<int> dset_ids(netcdf_variables_.size());
		std::vector<size_t> dims;
		netcdf_load_nd_dataset_helper(file_id, netcdf_variables_, dset_ids, min_dim, max_dim, dims, blob);
		
		//create start vector for Hyperslab-IO:
		std::vector<size_t> start(dims.size());
		unsigned long offset=1;
		for(unsigned int i=0; i<dims.size(); i++){
			offset*=dims[i];
			start[i]=0;
		}
		
		//read the data
		for(unsigned int i=0; i<netcdf_variables_.size(); i++){	
			int status = nc_get_vara_double(file_id, dset_ids[i], start.data(), dims.data(), &(blob->mutable_cpu_data()[i*offset]));
			CHECK_GT(status, 0) << "Failed to read double variable " << netcdf_variables_[i];
		}
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
		CHECK_GT(nc_inq_varid(loc_id, variable_name_.c_str(), &dset_id),0) << "Failed to find NetCDF variable " << variable_name_;
		
		// Get size of dataset
		char* buffer;
		int status = nc_get_var_string(loc_id, dset_id, &buffer);
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
		CHECK_GT(nc_inq_varid(loc_id, variable_name_.c_str(), &dset_id),0) << "Failed to find NetCDF variable " << variable_name_;
		
		int val;
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
	//	CHECK_GE(status, 0) << "Error while counting NetCDF links.";
	//	return info.nlinks;
	//}

	//string netcdf_get_name_by_idx(hid_t loc_id, int idx) {
	//	ssize_t str_size = H5Lget_name_by_idx(
	//		loc_id, ".", H5_INDEX_NAME, H5_ITER_NATIVE, idx, NULL, 0, H5P_DEFAULT);
	//	CHECK_GE(str_size, 0) << "Error retrieving NetCDF variable at index " << idx;
	//	char *c_str = new char[str_size+1];
	//	ssize_t status = H5Lget_name_by_idx(
	//		loc_id, ".", H5_INDEX_NAME, H5_ITER_NATIVE, idx, c_str, str_size+1,
	//	H5P_DEFAULT);
	//	CHECK_GE(status, 0) << "Error retrieving NetCDF variable at index " << idx;
	//	string result(c_str);
	//	delete[] c_str;
	//	return result;
	//}

}  // namespace caffe
#endif
