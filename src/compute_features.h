#ifndef COMPUTE_FEATURES_H_
#define COMPUTE_FEATURES_H_

#include "common.h"

#include <vector>
#include <string>

void map_pool5_features(std::vector<Frame> image_ptr, char* feature_ptr);

#endif // COMPUTE_FEATURES_H_
