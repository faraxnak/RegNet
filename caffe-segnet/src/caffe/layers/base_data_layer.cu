#include <vector>

#include "caffe/data_layers.hpp"

namespace caffe {

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, join the thread
  JoinPrefetchThread();
  // Reshape to loaded data.
  top[0]->ReshapeLike(this->prefetch_data_);
  // Copy the data
  caffe_copy(prefetch_data_.count(), prefetch_data_.cpu_data(),
      top[0]->mutable_gpu_data());
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(prefetch_label_);
    // Copy the labels.
    caffe_copy(prefetch_label_.count(), prefetch_label_.cpu_data(),
        top[1]->mutable_gpu_data());
  }
  if (this->output_labels_ && !this->output_floatImages_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(prefetch_label_);
    // Copy the labels.
    caffe_copy(prefetch_label_.count(), prefetch_label_.cpu_data(),
        top[1]->mutable_gpu_data());
  }
  if (this->output_floatImages_ && this->output_labels_ && !this->output_labels_y_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(prefetch_float_);
    // Copy the labels.
    caffe_copy(prefetch_float_.count(), prefetch_float_.cpu_data(),
        top[1]->mutable_gpu_data());
    // Reshape to loaded labels.
    top[2]->ReshapeLike(prefetch_label_);
    // Copy the labels.
    caffe_copy(prefetch_label_.count(), prefetch_label_.cpu_data(),
        top[2]->mutable_gpu_data());
  }
  if (this->output_floatImages_ && this->output_labels_ && this->output_labels_y_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(prefetch_float_);
    // Copy the labels.
    caffe_copy(prefetch_float_.count(), prefetch_float_.cpu_data(),
        top[1]->mutable_gpu_data());
    // Reshape to loaded labels.
    top[2]->ReshapeLike(prefetch_label_);
    // Copy the labels.
    caffe_copy(prefetch_label_.count(), prefetch_label_.cpu_data(),
        top[2]->mutable_gpu_data());
    // Reshape to loaded labels.
    top[3]->ReshapeLike(prefetch_label_y_);
    // Copy the labels.
    caffe_copy(prefetch_label_y_.count(), prefetch_label_y_.cpu_data(),
        top[3]->mutable_gpu_data());
  }
  // Start a new prefetch thread
  CreatePrefetchThread();
}

INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingDataLayer);

}  // namespace caffe
