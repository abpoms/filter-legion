#include "image_operations.h"
#include "halide/to_conv_patch.h"

int to_conv_input(Frame* in, Frame* out, Frame* mean) {
  buffer_t in_buffer = {0};
  in_buffer.host = reinterpret_cast<uint8_t*>(in->data);
  in_buffer.extent[0] = in->width;
  in_buffer.extent[1] = in->height;
  in_buffer.extent[2] = in->channels;
  in_buffer.stride[0] = in->channels;
  in_buffer.stride[1] = in->width * in->channels;
  in_buffer.stride[2] = 1;
  in_buffer.elem_size = in->element_size;

  buffer_t out_buffer = {0};
  out_buffer.host = reinterpret_cast<uint8_t*>(out->data);
  out_buffer.extent[0] = out->width;
  out_buffer.extent[1] = out->height;
  out_buffer.extent[2] = out->channels;
  out_buffer.stride[0] = 1;
  out_buffer.stride[1] = out->width;
  out_buffer.stride[2] = out->width * out->height;
  out_buffer.elem_size = out->element_size;

  buffer_t mean_buffer = {0};
  mean_buffer.host = reinterpret_cast<uint8_t*>(mean->data);
  mean_buffer.extent[0] = mean->width;
  mean_buffer.extent[1] = mean->height;
  mean_buffer.extent[2] = mean->channels;
  mean_buffer.stride[0] = 1;
  mean_buffer.stride[1] = mean->width;
  mean_buffer.stride[2] = mean->width * mean->height;
  mean_buffer.elem_size = mean->element_size;

  return ::to_conv_patch(&in_buffer,
                         &mean_buffer,
                         out->width, out->height,
                         &out_buffer);
}
