#ifndef COMPUTE_FEATURES_H_
#define COMPUTE_FEATURES_H_

#include "common.h"

#include <vector>
#include <string>

void init_neural_net();

void map_pool5_features(Frame* image_ptr, char* feature_ptr);

#endif // COMPUTE_FEATURES_H_
