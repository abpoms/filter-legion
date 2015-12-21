#ifndef COMMON_H_
#define COMMON_H_

const int VEC_DIM = 9216;

struct Frame {
  Frame()
  : width(0), height(0), channels(0), element_size(0), data(nullptr) {}

  Frame(int width, int height, int channels, int element_size)
  : width(width),
    height(height),
    channels(channels),
    element_size(element_size) {
    data = new char[width * height * channels * element_size];
  }

  int width;
  int height;
  int channels;
  int element_size;
  char* data;
};

#endif
