#include <vector>

#include "caffe/layers/yolo_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	template <typename Dtype>
	void YoloLossLayer<Dtype>::Reshape(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		LossLayer<Dtype>::Reshape(bottom, top);
		CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
			<< "Inputs must have the same dimension.";
		diff_.ReshapeLike(*bottom[0]);
		sum_.ReshapeLike(*bottom[0]);
		ratio_.ReshapeLike(*bottom[0]);
		prob_.ReshapeLike(*bottom[0]);
	}

	template <typename Dtype>
	void YoloLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
		//some global parameters
		int num = bottom[0]->num();
		int numfilters=bottom[0]->shape(1);
		int numclasses=numfilters-5;
		int numlabels = bottom[0]->shape(2);
		int ny = bottom[0]->shape(3);
		int nx = bottom[0]->shape(4);
		
		//determine some layer parameters:
		int gtid=this->layer_param_.yolo_loss_param().ground_truth_id();
		if(gtid!=0 || gtid!=1) DLOG(FATAL) << "Please specify 0 or 1 for the groung truth id!";
		Blob<Dtype>* gtblob = bottom[gtid];
		Dtype weight_center=this->layer_param_.yolo_loss_param().weight_center_loss();
		Dtype weight_size=this->layer_param_.yolo_loss_param().weight_size_loss();
		Dtype weight_xent=this->layer_param_.yolo_loss_param().weight_classification_loss();
		Dtype weight_confidence=this->layer_param_.yolo_loss_param().weight_confidence_loss();
		Dtype box_regulator=this->layer_param_.yolo_loss_param().box_regularization();

		//some useful variables:
		int chunksize=numlabels*nx*ny;
		int offset, offsetc, offset0;
		
		//define loss terms
		Dtype centerloss=Dtype(0);
		Dtype sizeloss=Dtype(0);
		Dtype confidenceloss=Dtype(0);
		Dtype xentloss=Dtype(0);
			
		//iterate over batches
		for(unsigned int n=0; n<num; ++n){
			//do the euclidian differences for xc, yc, dx, dy, conf and 
			//multiply the first four with the confidence of the ground-truth
			
			
			//------------------ CONFIDENCE ------------------
			//conf
			offsetc=chunksize*(4+n*numfilters);
			caffe_sub(chunksize,
					&(bottom[0]->cpu_data()[offsetc]),
					&(bottom[1]->cpu_data()[offsetc]),
					&(diff_.mutable_cpu_data()[offsetc]));
			//add to loss
			confidenceloss += caffe_cpu_dot(chunksize,&(diff_.cpu_data()[offsetc]),&(diff_.cpu_data()[offsetc]));
			
			
			//------------------ CENTER COORDS ------------------
			//xc
			offset=chunksize*(0+n*numfilters);
			//difference
			caffe_sub(chunksize,
					&(bottom[0]->cpu_data()[offset]),
					&(bottom[1]->cpu_data()[offset]),
					&(diff_.mutable_cpu_data()[offset]));
			//masking
			caffe_mul(chunksize, 
					&(gtblob->cpu_data()[offsetc]),
					&(diff_.mutable_cpu_data()[offset]), 
					&(diff_.mutable_cpu_data()[offset]));
			//add to loss:
			centerloss += caffe_cpu_dot(chunksize,&(diff_.cpu_data()[offset]),&(diff_.cpu_data()[offset]));
			//yc
			offset=chunksize*(1+n*numfilters);
			//difference
			caffe_sub(chunksize,
					&(bottom[0]->cpu_data()[offset]),
					&(bottom[1]->cpu_data()[offset]),
					&(diff_.mutable_cpu_data()[offset]));
			//masking
			caffe_mul(chunksize, 
					&(gtblob->cpu_data()[offsetc]),
					&(diff_.mutable_cpu_data()[offset]), 
					&(diff_.mutable_cpu_data()[offset]));
			//add to loss:
			centerloss += caffe_cpu_dot(chunksize,&(diff_.cpu_data()[offset]),&(diff_.cpu_data()[offset]));
			
			
			//------------------ BOX SIZE ------------------
			//this one is tricky as it needs regularization:
			//we save memory by putting the difference x-y, x+y in the data buffers of diff_ and sum_ respectively,
			//and put their corresponding squares into the diff-buffers of these blobs
			//w
			offset=chunksize*(2+n*numfilters);
			//difference: x0-y0
			caffe_sub(chunksize,
					&(bottom[0]->cpu_data()[offset]),
					&(bottom[1]->cpu_data()[offset]),
					&(diff_.mutable_cpu_data()[offset]));
			//square: (x0-y0)^2
			caffe_sqr(chunksize,
					&(diff_.cpu_data()[offset]),
					&(diff_.mutable_cpu_diff()[offset]));
			//sum: x0+y0
			caffe_add(chunksize,
					&(bottom[0]->cpu_data()[offset]),
					&(bottom[1]->cpu_data()[offset]),
					&(sum_.mutable_cpu_data()[offset]));
			//sqr: (x0+y0)^2
			caffe_sqr(chunksize,
					&(sum_.cpu_data()[offset]),
					&(sum_.mutable_cpu_diff()[offset]));
					
			//h
			offset=chunksize*(3+n*numfilters);
			//difference: x1-y1
			caffe_sub(chunksize,
					&(bottom[0]->cpu_data()[offset]),
					&(bottom[1]->cpu_data()[offset]),
					&(diff_.mutable_cpu_data()[offset]));
			//square: (x1-y1)^2
			caffe_sqr(chunksize,
					&(diff_.cpu_data()[offset]),
					&(diff_.mutable_cpu_diff()[offset]));
			//sum: x1+y1
			caffe_add(chunksize,
					&(bottom[0]->cpu_data()[offset]),
					&(bottom[1]->cpu_data()[offset]),
					&(sum_.mutable_cpu_data()[offset]));
			//sqr: (x1+y1)^2
			caffe_sqr(chunksize,
					&(sum_.cpu_data()[offset]),
					&(sum_.mutable_cpu_diff()[offset]));
					
			//combine: (x0-y0)^2 + (x1-y1)^2
			caffe_add(chunksize,
					&(diff_.cpu_diff()[offset-chunksize]),
					&(diff_.cpu_diff()[offset]),
					&(ratio_.mutable_cpu_data()[offset]));
			//masking: conf*( (x0-y0)^2 + (x1-y1)^2 )
			caffe_mul(chunksize, 
					&(gtblob->cpu_data()[offsetc]),
					&(ratio_.cpu_data()[offset]), 
					&(ratio_.mutable_cpu_data()[offset]));
					
			//combine: (x0+y0)^2+(x1+y1)^2
			caffe_add(chunksize,
					&(sum_.cpu_diff()[offset-chunksize]),
					&(sum_.cpu_diff()[offset]),
					&(ratio_.mutable_cpu_diff()[offset]));
			//regularization: (x0+y0)^2+(x1+y1)^2+reg
			caffe_add_scalar(chunksize, 
					box_regulator, 
					&(ratio_.mutable_cpu_diff()[offset]));
			
			//final combine step:
			//divide out: ||x-y||^2/||x+y||^2_reg
			caffe_div(chunksize,
					&(ratio_.cpu_data()[offset]),
					&(ratio_.cpu_diff()[offset]),
					&(ratio_.mutable_cpu_data()[offset]));
			//sum up:
			sizeloss += caffe_cpu_asum(chunksize,&(ratio_.cpu_data()[offset]));
			
			
			////------------------ CLASSIFICATION ------------------
			////perform softmax on the two blobs:
			////offset is universal:
			//offset0=chunksize*(5+n*numfilters);
			////first term:
			//caffe_exp(chunksize,
			//		&(bottom[0]->cpu_data()[offset0]),
			//		&(b0tmp_.mutable_cpu_data()[offset0]));
			//caffe_copy(chunksize, 
			//		&(b0tmp_.cpu_data()[offset0]), 
			//		&(prob_.mutable_cpu_data()[offset0]));
			//for(unsigned int c=1; c<numclasses; c++){
			//	offset=chunksize*(5+c+n*numfilters);
			//	caffe_exp(chunksize,
			//			&(bottom[0]->cpu_data()[offset]),
			//			&(b0tmp_.mutable_cpu_data()[offset]));
			//	caffe_add(chunksize,
			//			&(b0tmp_.cpu_data()[offset]),
			//			&(prob_.cpu_data()[offset0]),
			//			&(prob_.mutable_cpu_data()[offset0]));
			//}
			////normalize
			//for(unsigned int c=0; c<numclasses; c++){
			//	offset=chunksize*(5+c+n*numfilters);
			//	caffe_div(chunksize,
			//			&(b0tmp_.cpu_data()[offset]),
			//			&(prob_.cpu_data()[offset0]),
			//			&(b0tmp_.mutable_cpu_data()[offset]));
			//}
			//
			////second term:
			//caffe_exp(chunksize,
			//		&(bottom[1]->cpu_data()[offset0]),
			//		&(b1tmp_.mutable_cpu_data()[offset0]));
			//caffe_copy(chunksize, 
			//		&(b1tmp_.cpu_data()[offset0]), 
			//		&(prob_.mutable_cpu_data()[offset0]));
			//for(unsigned int c=1; c<numclasses; c++){
			//	offset=chunksize*(5+c+n*numfilters);
			//	caffe_exp(chunksize,
			//			&(bottom[1]->cpu_data()[offset]),
			//			&(b1tmp_.mutable_cpu_data()[offset]));
			//	caffe_add(chunksize,
			//			&(b1tmp_.cpu_data()[offset]),
			//			&(prob_.cpu_data()[offset0]),
			//			&(prob_.mutable_cpu_data()[offset0]));
			//}
			////normalize and log and then sum up:
			//for(unsigned int c=0; c<numclasses; c++){
			//	offset=chunksize*(5+c+n*numfilters);
			//	caffe_div(chunksize,
			//			&(b1tmp_.cpu_data()[offset]),
			//			&(prob_.cpu_data()[offset0]),
			//			&(b1tmp_.mutable_cpu_data()[offset]));
			//	//take log:
			//	caffe_log(chunksize, 
			//			&(b1tmp_.cpu_data()[offset]),
			//			&(b1tmp_.mutable_cpu_data()[offset]));
			//	//dotprod and sum up:
			//	xentloss -= caffe_cpu_dot(chunksize,&(b0tmp_.cpu_data()[offset]),&(b1tmp_.cpu_data()[offset]));
			//}
		}
		
		//normalize losses to allow for simplified gradients
		centerloss /= Dtype(2);
		confidenceloss /= Dtype(2);
		sizeloss /= Dtype(2);
		
		//DEBUG
		std::cout << "centerloss: " << centerloss << std::endl;
		std::cout << "confidenceloss: " << confidenceloss << std::endl;
		std::cout << "sizeloss: " << sizeloss << std::endl;
		std::cout << "xentloss: " << xentloss << std::endl;
		exit(EXIT_FAILURE);
		//DEBUG
		
		//combined loss:
		Dtype loss = weight_center*centerloss + weight_size*sizeloss + weight_confidence*confidenceloss + weight_xent*xentloss;
		top[0]->mutable_cpu_data()[0] = loss/Dtype(num);
	}

	template <typename Dtype>
	void YoloLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		
		//some global parameters
		int num = bottom[0]->num();
		int numfilters=bottom[0]->shape(1);
		int numclasses=numfilters-5;
		int numlabels = bottom[0]->shape(2);
		int ny = bottom[0]->shape(3);
		int nx = bottom[0]->shape(4);
		
		//determine some layer parameters:
		int gtid=this->layer_param_.yolo_loss_param().ground_truth_id();
		if(gtid!=0 || gtid!=1) DLOG(FATAL) << "Please specify 0 or 1 for the groung truth id!";
		Blob<Dtype>* gtblob = bottom[gtid];
		Dtype weight_center=this->layer_param_.yolo_loss_param().weight_center_loss();
		Dtype weight_size=this->layer_param_.yolo_loss_param().weight_size_loss();
		Dtype weight_xent=this->layer_param_.yolo_loss_param().weight_classification_loss();
		Dtype weight_confidence=this->layer_param_.yolo_loss_param().weight_confidence_loss();
		Dtype box_regulator=this->layer_param_.yolo_loss_param().box_regularization();
		
		//some useful variables:
		int chunksize=numlabels*nx*ny;
		int offset, offsetc, offset0;
		
		//iterate over blobs
		for (int i = 0; i < 2; ++i) {
			if(!propagate_down[i]) continue;
			const Dtype sign = ( (i == 0) ? 1 : -1 );
			//do not multiply the sign in just yet
			const Dtype alpha = top[0]->cpu_diff()[0] / bottom[i]->num ();
			
			if (propagate_down[i]) {
				//iterate over batches
				for(unsigned int n=0; n<num; ++n){
					//as before, do one after the other:
					
					
					//------------------ CONFIDENCE ------------------
					//conf
					offsetc=chunksize*(4+n*numfilters);
					caffe_cpu_axpby(chunksize,
									sign*alpha,
									&(diff_.cpu_data()[offsetc]),
									Dtype(0),
									&(bottom[i]->mutable_cpu_diff()[offsetc]));
					
					
					//------------------ CENTER COORDS ------------------
					//xc
					offset=chunksize*(0+n*numfilters);
					caffe_cpu_axpby(chunksize,
									sign*alpha,
									&(diff_.cpu_data()[offset]),
									Dtype(0),
									&(bottom[i]->mutable_cpu_diff()[offset]));
					//mask that term
					caffe_mul(chunksize, 
							&(gtblob->cpu_data()[offsetc]),
							&(bottom[i]->cpu_diff()[offset]), 
							&(bottom[i]->mutable_cpu_diff()[offset]));
					
					//yc
					offset=chunksize*(1+n*numfilters);
					caffe_cpu_axpby(chunksize,
									sign*alpha,
									&(diff_.cpu_data()[offset]),
									Dtype(0),
									&(bottom[i]->mutable_cpu_diff()[offset]));
					//mask that term
					caffe_mul(chunksize, 
							&(gtblob->cpu_data()[offsetc]),
							&(bottom[i]->cpu_diff()[offset]), 
							&(bottom[i]->mutable_cpu_diff()[offset]));
					
					
					//------------------ BOX SIZE ------------------
					//w
					offset=chunksize*(2+n*numfilters);
					//term1 w/o the aditional denominator: -(w0+w1)*loss(w0,w1)
					caffe_mul(chunksize,
							&(sum_.cpu_data()[offset]),
							&(ratio_.cpu_data()[offset]),
							&(bottom[i]->mutable_cpu_diff()[offset]));
					//term2 w/o the additional denominator: sign*(w0-w1) and multiply
					//the existing data with -1 and sum: sign*(w0-w1)-(w0+w1)*loss(w0,w1)
					caffe_cpu_axpby(chunksize,
									sign*alpha,
									&(diff_.cpu_data()[offset]),
									Dtype(-1),
									&(bottom[i]->mutable_cpu_diff()[offset]));
					//divide out the denominator:
					caffe_div(chunksize,
							&(bottom[i]->cpu_diff()[offset]),
							&(ratio_.cpu_diff()[offset+chunksize]), //be careful here, offsets need to be right
							&(bottom[i]->mutable_cpu_diff()[offset]));
					//mask out that term
					caffe_mul(chunksize, 
							&(gtblob->cpu_data()[offsetc]),
							&(bottom[i]->cpu_diff()[offset]), 
							&(bottom[i]->mutable_cpu_diff()[offset]));
					
					//h
					offset=chunksize*(3+n*numfilters);
					//term1 w/o the aditional denominator: (h0+h1)*loss(h0,h1)
					caffe_mul(chunksize,
							&(sum_.cpu_data()[offset]),
							&(ratio_.cpu_data()[offset]),
							&(bottom[i]->mutable_cpu_diff()[offset]));
					//term2 w/o the additional denominator: sign*(h0-h1) and multiply
					//the existing data with -1 and sum: sign*(h0-h1)-(h0+h1)*loss(h0,h1)
					caffe_cpu_axpby(chunksize,
									sign*alpha,
									&(diff_.cpu_data()[offset]),
									Dtype(-1),
									&(bottom[i]->mutable_cpu_diff()[offset]));
					//divide out the denominator:
					caffe_div(chunksize,
							&(bottom[i]->cpu_diff()[offset]),
							&(ratio_.cpu_diff()[offset]), //be careful here, offsets need to be right
							&(bottom[i]->mutable_cpu_diff()[offset]));
					//mask out that term
					caffe_mul(chunksize, 
							&(gtblob->cpu_data()[offsetc]),
							&(bottom[i]->cpu_diff()[offset]), 
							&(bottom[i]->mutable_cpu_diff()[offset]));
				}
			}
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(YoloLossLayer);
#endif

	INSTANTIATE_CLASS(YoloLossLayer);
	REGISTER_LAYER_CLASS(YoloLoss);

}  // namespace caffe
