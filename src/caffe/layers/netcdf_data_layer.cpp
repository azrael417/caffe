#ifdef USE_NETCDF
/*
TODO:
- load file in a separate thread ("prefetch")
- can be smarter about the memcpy call instead of doing it row-by-row
:: use util functions caffe_copy, and Blob->offset()
:: don't forget to update netcdf_daa_layer.cu accordingly
- add ability to shuffle filenames if flag is set
*/
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#include <netcdf.h>


#include "caffe/layers/netcdf_data_layer.hpp"
#include "caffe/util/netcdf.hpp"

namespace caffe {

	template <typename Dtype>
	NetCDFDataLayer<Dtype>::~NetCDFDataLayer<Dtype>() { }

	// Load data and label from netcdf filename into the class property blobs.
	template <typename Dtype>
	void NetCDFDataLayer<Dtype>::LoadNetCDFFileData(const char* filename) {
		int file_id;
	
		DLOG(INFO) << "Loading NetCDF file: " << filename;
		int retval = nc_open(filename, NC_NOWRITE, &file_id);
		if (retval != 0) {
			LOG(FATAL) << "Failed opening NetCDF file: " << filename;
		}
  
		int top_size = this->layer_param_.top_size();
		netcdf_blobs_.resize(top_size);

		const int MIN_DATA_DIM = 1;
		const int MAX_DATA_DIM = INT_MAX;

		for (int i = 0; i < top_size; ++i) {
			netcdf_blobs_[i] = shared_ptr<Blob<Dtype> >(new Blob<Dtype>());
			netcdf_load_nd_dataset(file_id, this->layer_param_.top(i).c_str(),
			MIN_DATA_DIM, MAX_DATA_DIM, netcdf_blobs_[i].get());
		}

		retval = nc_close(ncid);
		if (retval != 0) {
			LOG(FATAL) << "Failed to close NetCDF file: " << filename;
		}


		// MinTopBlobs==1 guarantees at least one top blob
		CHECK_GE(netcdf_blobs_[0]->num_axes(), 1) << "Input must have at least 1 axis.";
		const int num = netcdf_blobs_[0]->shape(0);
		for (int i = 1; i < top_size; ++i) {
			CHECK_EQ(netcdf_blobs_[i]->shape(0), num);
		}
		// Default to identity permutation.
		data_permutation_.clear();
		data_permutation_.resize(netcdf_blobs_[0]->shape(0));
		for (int i = 0; i < netcdf_blobs_[0]->shape(0); i++)
			data_permutation_[i] = i;

		// Shuffle if needed.
		if (this->layer_param_.netcdf_data_param().shuffle()) {
			std::random_shuffle(data_permutation_.begin(), data_permutation_.end());
			DLOG(INFO) << "Successully loaded " << netcdf_blobs_[0]->shape(0)
				<< " rows (shuffled)";
		} else {
			DLOG(INFO) << "Successully loaded " << netcdf_blobs_[0]->shape(0) << " rows";
		}
	}

	template <typename Dtype>
	void NetCDFDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
		// Refuse transformation parameters since HDF5 is totally generic.
		CHECK(!this->layer_param_.has_transform_param()) <<
			this->type() << " does not transform data.";
		// Read the source to parse the filenames.
		const string& source = this->layer_param_.netcdf_data_param().filelist();
		LOG(INFO) << "Loading list of NetCDF filenames from: " << source;
		netcdf_filenames_.clear();
		std::ifstream file_list(filelist.c_str());
		if (file_list.is_open()) {
			std::string line;
			while (file_list >> line) {
				netcdf_filenames_.push_back(line);
			}
		} else {
			LOG(FATAL) << "Failed to open source file: " << source;
		}
		file_list.close();
		num_files_ = netcdf_filenames_.size();
		current_file_ = 0;
		LOG(INFO) << "Number of NetCDF files: " << num_files_;
		CHECK_GE(num_files_, 1) << "Must have at least 1 NetCDF filename listed in " << source;
		
		//read list of netcdf variables which should be read from the file
		num_variables_ = this->layer_param_.netcdf_data_param().n_variables();
		for(unsigned int i=0; i< num_variables_; i++){
			netcdf_variables_.push_back(this->layer_param_.netcdf_data_param().variables()[i]);
		}
		LOG(INFO) << "Number of NetCDF variables: " << num_variables;
		CHECK_GE(num_variables_, 1) << "Must have at least 1 NetCDF variable listed in " << source;
		
		file_permutation_.clear();
		file_permutation_.resize(num_files_);
		// Default to identity permutation.
		for (int i = 0; i < num_files_; i++) {
			file_permutation_[i] = i;
		}

		// Shuffle if needed.
		if (this->layer_param_.netcdf_data_param().shuffle()) {
			std::random_shuffle(file_permutation_.begin(), file_permutation_.end());
		}

		// Load the first HDF5 file and initialize the line counter.
		LoadNetCDFFileData(hdf_filenames_[file_permutation_[current_file_]].c_str());
		current_row_ = 0;

		// Reshape blobs.
		const int batch_size = this->layer_param_.netcdf_data_param().batch_size();
		const int top_size = this->layer_param_.top_size();
		vector<int> top_shape;
		for (int i = 0; i < top_size; ++i) {
			top_shape.resize(netcdf_blobs_[i]->num_axes());
			top_shape[0] = batch_size;
			for (int j = 1; j < top_shape.size(); ++j) {
				top_shape[j] = netcdf_blobs_[i]->shape(j);
			}
			top[i]->Reshape(top_shape);
		}
		
		for(unsigned int i=0; i<netcdf_filenames_.size(); i++){
			std::cout << netcdf_filenames_[i] << std::endl;
		}
		for(unsigned int i=0; i<netcdf_variables_.size(); i++){
			std::cout << netcdf_variables_[i] << std::endl;
		}
		exit EXIT_FAILURE;
	}

	template <typename Dtype>
	void NetCDFDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
		const int batch_size = this->layer_param_.netcdf_data_param().batch_size();
		for (int i = 0; i < batch_size; ++i, ++current_row_) {
			if (current_row_ == netcdf_blobs_[0]->shape(0)) {
				if (num_files_ > 1) {
					++current_file_;
					if (current_file_ == num_files_) {
						current_file_ = 0;
						if (this->layer_param_.netcdf_data_param().shuffle()) {
							std::random_shuffle(file_permutation_.begin(),
							file_permutation_.end());
						}
						DLOG(INFO) << "Looping around to first file.";
					}
					LoadNetCDFFileData(
						netcdf_filenames_[file_permutation_[current_file_]].c_str());
				}
				current_row_ = 0;
				if (this->layer_param_.netcdf_data_param().shuffle())
					std::random_shuffle(data_permutation_.begin(), data_permutation_.end());
			}
			for (int j = 0; j < this->layer_param_.top_size(); ++j) {
				int data_dim = top[j]->count() / top[j]->shape(0);
				caffe_copy(data_dim,
				&netcdf_blobs_[j]->cpu_data()[data_permutation_[current_row_]
					* data_dim], &top[j]->mutable_cpu_data()[i * data_dim]);
			}
		}
	}

#ifdef CPU_ONLY
	STUB_GPU_FORWARD(NetCDFDataLayer, Forward);
#endif

	INSTANTIATE_CLASS(NetCDFDataLayer);
	REGISTER_LAYER_CLASS(NetCDFData);

}  // namespace caffe
#endif