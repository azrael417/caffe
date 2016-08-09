#include <vector>
#include <map>

#include "caffe/layers/box_to_yolo_layer.hpp"


namespace caffe {

template <typename Dtype>
void BoxToYoloLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
										const vector<Blob<Dtype>*>& top) 
{
	//some parameters
	unsigned int top_size=bottom.size();
	
	//some layer parameters
	bool masked = this->layer_param_.box_to_yolo_param().masked();
	unsigned int bstart=(masked ? 1 : 0);
	unsigned int odimx = this->layer_param_.box_to_yolo_param().orig_dimx();
	unsigned int odimy = this->layer_param_.box_to_yolo_param().orig_dimy();
	unsigned int rdimx = this->layer_param_.box_to_yolo_param().reduced_dimx();
	unsigned int rdimy = this->layer_param_.box_to_yolo_param().reduced_dimy();
	unsigned int rbx = static_cast<int>(floor(static_cast<double>(odimx/rdimx)));
	unsigned int rby = static_cast<int>(floor(static_cast<double>(odimy/rdimy)));
	unsigned int mlps = this->layer_param_.box_to_yolo_param().max_labels_per_segment();
	
	//class related stuff
	bool onehot = this->layer_param_.box_to_yolo_param().one_hot();
	unsigned int numclasses=1;
	//if one hot encoding is used, read in keys for classes
	std::map<int,int> classmap;
	if(onehot){
		numclasses=this->layer_param_.box_to_yolo_param().class_label_size();
		for(unsigned int c=0; c<numclasses; c++){
			classmap[this->layer_param_.box_to_yolo_param().class_label(c)]=c;
		}
	}
	
	//is the subdivided image smaller?
	if(rdimx>odimx || rdimy>odimy) DLOG(FATAL) << "The size of the subdivided image cannot be larger than the original image";
	
	//get some dimensions of the bottom blob:
	unsigned int batch_size=bottom[0]->shape(0);
	//max labels per image
	unsigned int mlpi=bottom[0]->shape(1);
	//number of channels
	unsigned int numvars=bottom[0]->shape(2);
	
	//some sanity checks:
	if( (masked && numvars!=5) ){
		DLOG(FATAL) << "Please specify 5 columns (mask, xmin, xmax, ymin, ymax).";
	}
	else if( (!masked && numvars!=4) ){
		DLOG(FATAL) << "Please specify 4 columns (xmin, xmax, ymin, ymax).";
	}
	
	//determine size of output:
	unsigned int yolosize=batch_size*rdimx*rdimy*mlps*(5+numclasses);
	
	//create output blob size:
	vector<int> top_shape;
	for (int i = 0; i < top_size; ++i) {
		top_shape.resize(5);
		top_shape[0] = batch_size;
		top_shape[1] = rdimx;
		top_shape[2] = rdimy;
		top_shape[3] = mlps;
		top_shape[4] = (5+numclasses);
		
		//reshape output
		top[i]->Reshape(top_shape);
	}
	unsigned int data_size=top[0]->count();
	
	//perform conversion:
	for (int i = 0; i < top_size; ++i) {
		//initialize everything to zero
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for(unsigned int d=0; d<data_size; d++) top[0]->mutable_cpu_data()[d]=0.;
	
		//perform the actual conversion
#ifdef _OPENMP
#pragma omp parallel for firstprivate(classmap) schedule(dynamic)
#endif
		for(unsigned int b=0; b<batch_size; b++){
			std::vector<int> bindex(3);
			bindex[0]=b;
			
			//hashmap to determine if cell was filled
			std::map<int, int> fillcount;
			
			for(unsigned int l=0; l<mlpi; l++){
				
				//image label
				bindex[2]=l;
				
				//skip if masked and mask is zero
				unsigned int bstart=0;
				if(masked){
					//get mask
					bindex[1]=0;
					if(bottom[0]->data_at(bindex)<1.e-8) break;
					bstart=1;
				}

				//get xmin, ymin, xmax, ymax
				bindex[1]=bstart;
				Dtype xmin=bottom[0]->data_at(bindex);
				bindex[1]=bstart+1;
				Dtype xmax=bottom[0]->data_at(bindex);
				bindex[1]=bstart+2;
				Dtype ymin=bottom[0]->data_at(bindex);
				bindex[1]=bstart+3;
				Dtype ymax=bottom[0]->data_at(bindex);
				
				//class:
				bindex[1]=bstart+4;
				int classid=static_cast<int>(bottom[0]->data_at(bindex));
				
				//compute center and box size:
				Dtype xcenter=0.5*(xmin+xmax);
				Dtype ycenter=0.5*(ymin+ymax);
				Dtype dx=(xmax-xmin);
				Dtype dy=(ymax-ymin);
				
				//determine quadrant, i.e. reduced center and offset:
				Dtype orx=static_cast<Dtype>(static_cast<int>(xcenter)%rbx);
				Dtype crx=static_cast<Dtype>(static_cast<int>(xcenter-orx)/rbx);
				Dtype ory=static_cast<Dtype>(static_cast<int>(ycenter)%rby);
				Dtype cry=static_cast<Dtype>(static_cast<int>(ycenter-ory)/rby);
				
				//determine box size in relation to big image
				Dtype drx=dx/static_cast<Dtype>(odimx);
				Dtype dry=dy/static_cast<Dtype>(odimy);
				
				//construct unique key:
				int key=static_cast<int>(crx)+rdimx*static_cast<int>(cry);
				if( fillcount.find(key)==fillcount.end() ){
					fillcount[key]=0;
				}
				else{
					fillcount[key]++;
					if(fillcount[key]>mlps){
						DLOG(FATAL) << "Expected maximally " << mlps << " labels per segment but got " << fillcount[key] << ". Increase the number of labels you expect by increasing the max_labels_per_segment variable.";
					}
				}
				
				//insert into tensor:
				std::vector<int> tindex(5);
				tindex[0]=b;
				tindex[1]=static_cast<int>(crx);
				tindex[2]=static_cast<int>(cry);
				tindex[3]=fillcount[key];
				
				//write the components
				int offset;
				//offset x:
				tindex[4]=0;
				offset=top[0]->offset(tindex);
				top[0]->mutable_cpu_data()[offset]=orx;
				//offset y
				tindex[4]=1;
				offset=top[0]->offset(tindex);
				top[0]->mutable_cpu_data()[offset]=ory;
				//width
				tindex[4]=2;
				offset=top[0]->offset(tindex);
				top[0]->mutable_cpu_data()[offset]=drx;
				//height
				tindex[4]=3;
				offset=top[0]->offset(tindex);
				top[0]->mutable_cpu_data()[offset]=dry;
				//confidence:
				tindex[4]=4;
				offset=top[0]->offset(tindex);
				top[0]->mutable_cpu_data()[offset]=1.;
				//class, one-hot encoded:
				tindex[4]=5+classmap[classid];
				offset=top[0]->offset(tindex);
				top[0]->mutable_cpu_data()[offset]=1.;
			}
		}
	}
}

template <typename Dtype>
void BoxToYoloLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
										const vector<bool>& propagate_down, 
										const vector<Blob<Dtype>*>& bottom) 
{
  if (!propagate_down[0]) { return; }
}

template <typename Dtype>
void BoxToYoloLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	//some parameters
	unsigned int top_size=bottom.size();
	
	//some layer parameters
	bool masked = this->layer_param_.box_to_yolo_param().masked();
	unsigned int rdimx = this->layer_param_.box_to_yolo_param().reduced_dimx();
	unsigned int rdimy = this->layer_param_.box_to_yolo_param().reduced_dimy();
	unsigned int mlps = this->layer_param_.box_to_yolo_param().max_labels_per_segment();
	
	//class related stuff
	bool onehot = this->layer_param_.box_to_yolo_param().one_hot();
	unsigned int numclasses=1;
	//if one hot encoding is used, read in keys for classes
	std::map<int,int> classmap;
	if(onehot){
		numclasses=this->layer_param_.box_to_yolo_param().class_label_size();
	}
	
	//get some dimensions of the bottom blob:
	unsigned int batch_size=bottom[0]->shape(0);
	
	//create output blob size:
	vector<int> top_shape;
	for (int i = 0; i < top_size; ++i) {
		top_shape.resize(5);
		top_shape[0] = batch_size;
		top_shape[1] = rdimx;
		top_shape[2] = rdimy;
		top_shape[3] = mlps;
		top_shape[4] = (5+numclasses);
		
		//reshape output
		top[i]->Reshape(top_shape);
	}
}


#ifdef CPU_ONLY
STUB_GPU(BoxToYoloLayer);
#endif

INSTANTIATE_CLASS(BoxToYoloLayer);
REGISTER_LAYER_CLASS(BoxToYolo);

}  // namespace caffe

