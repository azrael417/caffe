#ifndef CAFFE_BOXTOYOLO_LAYER_HPP_
#define CAFFE_BOXTOYOLO_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

	/**
	* @brief Turns a conventional box-label blob with (numbatch, image_id, xmin, xmax, ymin, ymax, class (mask)) 
	*        into a YOLO label with (numbatch, numimages, max_labels_per_patch, xcent, ycent, width, height, confidence, classvec )
	*        where classvec is one-hot-encoded
	*/
	template <typename Dtype>
	class BoxToYoloLayer : public Layer<Dtype> {
	public:
		explicit BoxToYoloLayer(const LayerParameter& param) : Layer<Dtype>(param) {}
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "BoxToYolo"; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	};

}  // namespace caffe

#endif  // CAFFE_BOXTOYOLO_LAYER_HPP_
