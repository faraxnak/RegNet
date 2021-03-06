#include <string>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/util/io.hpp"
//#include "caffe/layers/base_data_layer.hpp"

namespace caffe {

template <typename Dtype>
BaseDataLayer<Dtype>::BaseDataLayer(const LayerParameter& param)
    : Layer<Dtype>(param),
      transform_param_(param.transform_param()) {
}

template <typename Dtype>
void BaseDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (top.size() == 1) {
    output_labels_ = false;
    output_floatImages_  = false;
    output_labels_y_  = false;
  } else if (top.size() == 3) {
    output_floatImages_ = true;
    output_labels_ = true;
    output_labels_y_  = false;
  } else if (top.size() == 4) {
    output_labels_ = true;
    output_labels_y_ = true;
    output_floatImages_ = true;
  } else {
    output_labels_ = true;
    output_floatImages_ = false;
  }

  data_transformer_.reset(
      new DataTransformer<Dtype>(transform_param_, this->phase_));
  data_transformer_->InitRand();
  // The subclasses should setup the size of bottom and top
  DataLayerSetUp(bottom, top);
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BaseDataLayer<Dtype>::LayerSetUp(bottom, top);
  // Now, start the prefetch thread. Before calling prefetch, we make two
  // cpu_data calls so that the prefetch thread does not accidentally make
  // simultaneous cudaMalloc calls when the main thread is running. In some
  // GPUs this seems to cause failures if we do not so.
  this->prefetch_data_.mutable_cpu_data();
  if (this->output_labels_) {
    this->prefetch_label_.mutable_cpu_data();
  }
  if (this->output_floatImages_) {
    this->prefetch_float_.mutable_cpu_data();
  }
  if (this->output_labels_y_) {
    this->prefetch_label_y_.mutable_cpu_data();
  }
  DLOG(INFO) << "Initializing prefetch";
  this->CreatePrefetchThread();
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::CreatePrefetchThread() {
  this->data_transformer_->InitRand();
  CHECK(StartInternalThread2()) << "Thread execution failed";
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::JoinPrefetchThread() {
  CHECK(WaitForInternalThreadToExit()) << "Thread joining failed";
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, join the thread
  JoinPrefetchThread();
  DLOG(INFO) << "Thread joined";
  // Reshape to loaded data.
  top[0]->ReshapeLike(prefetch_data_);
  // Copy the data
  caffe_copy(prefetch_data_.count(), prefetch_data_.cpu_data(),
             top[0]->mutable_cpu_data());
  DLOG(INFO) << "Prefetch copied";
  if (this->output_labels_ && !this->output_floatImages_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(prefetch_label_);
    // Copy the labels.
    caffe_copy(prefetch_label_.count(), prefetch_label_.cpu_data(),
               top[1]->mutable_cpu_data());
  }
  if (this->output_floatImages_ && this->output_labels_ && !this->output_labels_y_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(prefetch_float_);
    // Copy the labels.
    caffe_copy(prefetch_float_.count(), prefetch_float_.cpu_data(),
               top[1]->mutable_cpu_data());
    // Reshape to loaded labels.
    top[2]->ReshapeLike(prefetch_label_);
    // Copy the labels.
    caffe_copy(prefetch_label_.count(), prefetch_label_.cpu_data(),
               top[2]->mutable_cpu_data());
  }
  if (this->output_floatImages_ && this->output_labels_ && this->output_labels_y_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(prefetch_float_);
    // Copy the labels.
    caffe_copy(prefetch_float_.count(), prefetch_float_.cpu_data(),
               top[1]->mutable_cpu_data());
    // Reshape to loaded labels.
    top[2]->ReshapeLike(prefetch_label_);
    // Copy the labels.
    caffe_copy(prefetch_label_.count(), prefetch_label_.cpu_data(),
               top[2]->mutable_cpu_data());
    //
    top[3]->ReshapeLike(prefetch_label_y_);
    //
    caffe_copy(prefetch_label_y_.count(), prefetch_label_y_.cpu_data(),
                top[3]->mutable_cpu_data());
  }
  // Start a new prefetch thread
  DLOG(INFO) << "CreatePrefetchThread";
  CreatePrefetchThread();
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(BasePrefetchingDataLayer, Forward);
#endif

INSTANTIATE_CLASS(BaseDataLayer);
INSTANTIATE_CLASS(BasePrefetchingDataLayer);

}  // namespace caffe
