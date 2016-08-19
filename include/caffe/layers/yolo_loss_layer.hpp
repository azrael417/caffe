#ifndef CAFFE_YOLO_LOSS_LAYER_HPP_
#define CAFFE_YOLO_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

	/**
	* @brief Computes the Yolo loss @f$
	*          E = \frac{1}{2N} \sum\limits_{n=1}^N \left| \left| \hat{y}_n - y_n
	*        \right| \right|_2^2 @f$ for real-valued regression tasks.
	*
	*/
	template <typename Dtype>
	class YoloLossLayer : public LossLayer<Dtype> {
	public:
		explicit YoloLossLayer(const LayerParameter& param)
			: LossLayer<Dtype>(param), diff_() {}
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "YoloLoss"; }
		/**
		* Unlike most loss layers, in the YoloLossLayer we can backpropagate
		* to both inputs -- override to return true and always allow force_backward.
		*/
		virtual inline bool AllowForceBackward(const int bottom_index) const {
			return true;
		}

	protected:
		/// @copydoc YoloLossLayer
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

		/**
		* @brief Computes the Yolo error gradient w.r.t. the inputs.
		*
		*/
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		//we need some backup blobs for computing the softmax part
		Blob<Dtype> diff_, sum_, ratio_, prob_;
	};

}  // namespace caffe

#endif  // CAFFE_YOLO_LOSS_LAYER_HPP_
