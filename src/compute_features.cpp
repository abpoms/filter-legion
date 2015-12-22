#include "compute_features.h"
#include "image_operations.h"

#include "jpeg/JPEGReader.h"
#include "jpeg/JPEGWriter.h"

// Caffe
#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

using caffe::Blob;
using caffe::BlobProto;
using caffe::Caffe;
using caffe::Net;
using boost::shared_ptr;
using std::string;

const int DIM = 227;

shared_ptr<Net<float>> init_neural_net(Frame mean, int batch_size) {
  std::string model_path =
    "features/hybridCNN/hybridCNN_deploy_upgraded.prototxt";
  std::string model_weights_path =
    "features/hybridCNN/hybridCNN_iter_700000_upgraded.caffemodel";
    //"features/places205VGG16/snapshot_iter_765280.caffemodel";
  std::string mean_proto_path =
    "features/hybridCNN/hybridCNN_mean.binaryproto";

  // Initialize our network
  shared_ptr<Net<float>> feature_extraction_net =
    shared_ptr<Net<float>>(new Net<float>(model_path, caffe::TEST));
  feature_extraction_net->CopyTrainedLayersFrom(model_weights_path);
  const shared_ptr<Blob<float>> data_blob =
    feature_extraction_net->blob_by_name("data");
  data_blob->Reshape({batch_size, 3, DIM, DIM});

  // Load mean image
  Blob<float> data_mean;
  BlobProto blob_proto;
  bool result = ReadProtoFromBinaryFile(mean_proto_path, &blob_proto);
  (void)result;
  data_mean.FromProto(blob_proto);
  memcpy(mean.data, data_mean.cpu_data(), sizeof(float) * 256 * 256 * 3);

  // MIT Places VGG-16 mean image
  // for (int i = 0; i < 224 * 224; ++i) {
  //   mean.data[i + (224 * 224) * 0] = 105.487823486f;
  //   mean.data[i + (224 * 224) * 1] = 113.741088867f;
  //   mean.data[i + (224 * 224) * 2] = 116.060394287f;
  // }
  return feature_extraction_net;
}

void map_pool5_features(std::vector<Frame> frames, char* features_ptr) {
  Frame mean(256, 256, 3, sizeof(float));

  int num_images = frames.size();

  int BATCH_SIZE = 16;
  shared_ptr<Net<float> > feature_extraction_net =
    init_neural_net(mean, BATCH_SIZE);

  const shared_ptr<Blob<float>> data_blob =
    feature_extraction_net->blob_by_name("data");


  Blob<float> input{BATCH_SIZE, 3, DIM, DIM};

  for (int i = 0; i < num_images; i+=BATCH_SIZE) {
    int current_batch = BATCH_SIZE;
    if (current_batch + i > num_images) {
      current_batch = num_images - i;
      data_blob->Reshape({current_batch, 3, DIM, DIM});
      input.Reshape({current_batch, 3, DIM, DIM});
    }
    float *data = input.mutable_cpu_data();

    // Pack image into blob
    Frame conv_input(DIM, DIM, 3, sizeof(float));
    for (int j = 0; j < current_batch; ++j) {
      Frame image_ptr = frames[i + j];
      to_conv_input(&image_ptr, &conv_input, &mean);
      memcpy(data + (DIM * DIM * 3) * j,
             conv_input.data, DIM * DIM * 3 * sizeof(float));
    }
    delete[] conv_input.data;

    // Evaluate network
    feature_extraction_net->Forward({&input});

    // Extract features from Blob
    // TODO(abp): I can probably just allocate my own giant set of memory,
    // use it as the output for the net, and then create a bunch of images
    // backed by the features output into that memory
    const shared_ptr<Blob<float>> features_data =
      feature_extraction_net->blob_by_name("pool5");

    memcpy(features_ptr + i * VEC_DIM,
           features_data->cpu_data(),
           sizeof(float) * VEC_DIM * BATCH_SIZE);
  }

  delete[] mean.data;
}
